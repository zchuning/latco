import os
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

import planning_agent
from planners import gn_solver
from utils import tools


class LatCo(planning_agent.PlanningAgent):
  def pair_residual_func_body(self, x_a, x_b, lam, nu):
    """ This function is required by the gn_solver. It taking in the pair of adjacent states (current and next state)
    and outputs the residuals for the Gauss-Newton optimization
  
    :param x_a: current states and actions, batched across sequence length
    :param x_b: next states and actions
    :param lam: lagrange multiplier for dynamics
    :param nu: lagrange multiplier for actions
    :return: a vector of residuals
    """
    # Compute residuals
    actions_a = x_a[:, -self._actdim:][None]
    feats_a = x_a[:, :-self._actdim][None]
    states_a = self._dynamics.from_feat(feats_a)
    prior_a = self._dynamics.img_step(states_a, actions_a)
    x_b_pred = self._dynamics.get_mean_feat(prior_a)[0]
    dyn_residual = x_b[:, :-self._actdim] - x_b_pred
    act_residual = tf.clip_by_value(tf.math.abs(x_a[:, -self._actdim:]) - 1, 0, np.inf)
    rew = self._reward(x_b[:, :-self._actdim]).mode()[:, None]
    rew_residual = tf.math.softplus(-rew)

    # Compute coefficients
    dyn_c = tf.sqrt(lam)[:, :, None] * self._c.dyn_loss_scale
    act_c = tf.sqrt(nu)[:, :, None] * self._c.act_loss_scale
    rew_c = tf.ones(lam.shape, np.float32)[:, :, None]

    # Normalize with the sum of multipliers to scale the objective in a reasonable range.
    bs, n = nu.shape[0:2]
    normalize = 1 / (tf.reduce_mean(dyn_c, 1) + tf.reduce_mean(act_c, 1) + tf.reduce_mean(rew_c, 1))
    dyn_resw = dyn_c * tf.reshape(dyn_residual, (bs, n, -1))
    act_resw = act_c * tf.reshape(act_residual, (bs, n, -1))
    rew_resw = rew_c * tf.reshape(rew_residual, (bs, n, -1))
    objective = normalize[:, :, None] * tf.concat([dyn_resw, act_resw, rew_resw], 2)

    return tf.reshape(objective, (-1, objective.shape[2]))

  @tf.function
  def opt_step(self, plan, init_feat, lam, nu):
    """ One optimization step. This function is needed for the code to compile properly """
    # We actually also optimize the first state, ensuring it is close to the true first state
    init_residual_func = lambda x: (x[:, :-self._actdim] - init_feat) * 1000
    pair_residual_func = lambda x_a, x_b : self.pair_residual_func_body(x_a, x_b, lam, nu)
    plan = gn_solver.solve_step(pair_residual_func, init_residual_func, plan, damping=self._c.gn_damping)
    return plan

  def _plan(self, init_obs, save_images, step, init_feat=None, verbose=True):
    """ The LatCo agent. This function implements the dual descent algorithm. _batch_ optimization procedures are
    executed in parallel, and the best solution is taken.
    
    :param init_obs: Initial observation (either observation of latent has to be specified)
    :param save_images: Whether to save images
    :param step: Index to label the saved images with
    :param init_feat: Initial latent state (either observation of latent has to be specified)
    """
    hor = self._c.planning_horizon
    feat_size = self._c.stoch_size + self._c.deter_size
    var_len_step = feat_size + self._actdim
    batch = self._c.n_parallel_plans
    dyn_threshold = self._c.dyn_threshold
    act_threshold = self._c.act_threshold

    if init_feat is None:
      init_feat, _ = self.get_init_feat(init_obs)
    plan = tf.random.normal((batch, (hor + 1) * var_len_step,), dtype=self._float)
    # Set the first state to be the observed initial state
    plan = tf.concat([tf.repeat(init_feat, batch, 0), plan[:, feat_size:]], 1)
    plan = tf.reshape(plan, [batch, hor + 1, var_len_step])
    lam = tf.ones((batch, hor)) * self._c.init_lam
    nu = tf.ones((batch, hor)) * self._c.init_nu

    # Run dual descent
    plans = [plan]
    metrics = tools.AttrDefaultDict(list)
    for i in range(self._c.optimization_steps):
      # Run Gauss-Newton step
      plan = self.opt_step(plan, init_feat, lam, nu)
      plan_res = tf.reshape(plan, [batch, hor+1, -1])
      feat_preds, act_preds = tf.split(plan_res, [feat_size, self._actdim], 2)
      states = self._dynamics.from_feat(feat_preds[:, :-1])
      priors = self._dynamics.img_step(states, act_preds[:, :-1])
      priors_feat = tf.squeeze(self._dynamics.get_mean_feat(priors))
      dyn_viol = tf.reduce_sum(tf.square(priors_feat - feat_preds[:, 1:]), 2)
      act_viol = tf.reduce_sum(tf.clip_by_value(tf.square(act_preds[:, :-1]) - 1, 0, np.inf), 2)

      # Update lagrange multipliers
      if i % self._c.lm_update_every == self._c.lm_update_every - 1:
        lam_delta = lam * 0.1 * tf.math.log((dyn_viol + 0.1 * dyn_threshold) / dyn_threshold) / tf.math.log(10.0)
        nu_delta  = nu * 0.1 * tf.math.log((act_viol + 0.1 * act_threshold) / act_threshold) / tf.math.log(10.0)
        lam = lam + lam_delta
        nu = nu + nu_delta

      # Logging
      act_preds_clipped = tf.clip_by_value(act_preds, -1, 1)
      metrics.dynamics.append(tf.reduce_sum(dyn_viol))
      metrics.action_violation.append(tf.reduce_sum(act_viol))
      metrics.dynamics_coeff.append(self._c.dyn_loss_scale**2 * tf.reduce_sum(lam))
      metrics.action_coeff.append(self._c.act_loss_scale**2 * tf.reduce_sum(nu))
      plans.append(plan)

      if self._c.log_colloc_scalars:
        # Compute and record dynamics loss and reward
        rew_raw = self._reward(feat_preds).mode()
        metrics.rewards.append(tf.reduce_sum(rew_raw, 1))

        # Record model rewards
        model_feats = self._dynamics.imagine_feat(act_preds_clipped[0:1], init_feat, deterministic=True)
        model_rew = self._reward(model_feats[0:1]).mode()
        metrics.model_rewards.append(tf.reduce_sum(model_rew))

    # Select best plan
    model_feats = self._dynamics.imagine_feat(act_preds_clipped, tf.repeat(init_feat, batch, 0), deterministic=False)
    model_rew = tf.reduce_sum(self._reward(model_feats).mode(), [1])
    best_plan = tf.argmax(model_rew)
    predicted_rewards = model_rew[best_plan]
    metrics.predicted_rewards.append(predicted_rewards)

    # Get action and feature predictions
    act_preds = act_preds[best_plan, :min(hor, self._c.mpc_steps)]
    if tf.reduce_any(tf.math.is_nan(act_preds)) or tf.reduce_any(tf.math.is_inf(act_preds)):
      act_preds = tf.zeros_like(act_preds)
    feat_preds = feat_preds[best_plan, :min(hor, self._c.mpc_steps)]
    if self._c.log_colloc_scalars:
      metrics.rewards = [r[best_plan] for r in metrics.rewards]
    else:
      metrics.rewards = [tf.reduce_sum(self._reward(feat_preds).mode())]

    # Logging
    img_preds = None
    if save_images:
      img_preds = self._decode(feat_preds).mode()
      self.logger.log_graph('losses', {f'{c[0]}/{step}': c[1] for c in metrics.items()})
      self.visualize_colloc(img_preds, act_preds, init_feat, step)
    if verbose:
      if batch > 1:
        print(f'plan rewards: {model_rew}, best plan: {best_plan}')
      print(f"Planned average dynamics loss: {metrics.dynamics[-1] / hor}")
      print(f"Planned average action violation: {metrics.action_violation[-1] / hor}")
      print(f"Planned total reward: {metrics.predicted_rewards[-1] / hor}")
    info = {'metrics': tools.map_dict(lambda x: x[-1] / hor if len(x) > 0 else 0, dict(metrics)),
            'plans': tf.stack(plans, 0)[:, best_plan:best_plan + 1],
            'curves': dict(metrics)}
    return act_preds, img_preds, feat_preds, info
