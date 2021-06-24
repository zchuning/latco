import argparse
import imageio
import os
import pathlib
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'
sys.path.append(str(pathlib.Path(__file__).parent))

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
from tensorflow_probability import distributions as tfd

tf.get_logger().setLevel('ERROR')

from planners import gn_solver
from latco import LatCo
from utils import tools


class ProbabilisticLatCo(LatCo):
  def pair_residual_func_body(self, x_a, x_b,
      lam=np.ones(1, np.float32), nu=np.ones(1, np.float32)):

    n_problatco_samples = self._c.n_problatco_samples # particles to use in moment matching
    # Compute residuals
    feat_size = self._c.stoch_size + self._c.deter_size
    actions_a = x_a[:, -self._actdim:][None]
    feats_a = x_a[:, :feat_size]
    feats_a_std = tf.stop_gradient((x_a[:, feat_size:-self._actdim]))
    feats_a_sample = tf.random.normal([n_problatco_samples] + list(feats_a.shape), feats_a, feats_a_std)
    states_a = self._dynamics.from_feat(feats_a_sample)
    feats_b = x_b[:, :feat_size]
    feats_b_std = ((x_b[:, feat_size:-self._actdim]))
    feats_b_sample = tf.random.normal(feats_b.shape, feats_b, tf.stop_gradient(feats_b_std))

    # Predict
    state_b_pred = self._dynamics.img_step(states_a, tf.repeat(actions_a, n_problatco_samples, 0))
    feats_b_pred = self._dynamics.get_feat(state_b_pred)
    feats_b_pred_mean = tf.reduce_mean(feats_b_pred, 0)
    feats_b_pred_std = tf.math.reduce_std(feats_b_pred, 0)

    # Dynamics residual
    dyn_res = feats_b - feats_b_pred_mean
    dyn_res_std = (feats_b_std - tf.stop_gradient(feats_b_pred_std))
    dyn_res = tf.concat([dyn_res, dyn_res_std], 1)
    # Action residual
    act_res = tf.clip_by_value(tf.math.abs(x_a[:, -self._actdim:]) - 1, 0, np.inf)
    # Reward residual
    rew = self._reward(feats_b_sample).mode()[:, None]
    rew_res = tf.math.softplus(-rew)

    # Compute coefficients
    dyn_c = tf.sqrt(lam)[:, :, None]
    act_c = tf.sqrt(nu)[:, :, None]
    rew_c = tf.ones((1, np.float32, 1))

    # Normalize each plan in the batch independently
    bs, n = nu.shape[0:2]
    normalize = 1 / (tf.reduce_mean(dyn_c, 1) + tf.reduce_mean(act_c, 1) + tf.reduce_mean(rew_c, 1))
    dyn_resw = dyn_c * tf.reshape(dyn_res, (bs, n, -1))
    act_resw = act_c * tf.reshape(act_res, (bs, n, -1))
    rew_resw = rew_c * tf.reshape(rew_res, (bs, n, -1))
    objective = normalize[:, :, None] * tf.concat([dyn_resw, act_resw, rew_resw], 2)

    return tf.reshape(objective, (-1, objective.shape[2])), dyn_res, act_res, rew_res, dyn_resw, act_resw, rew_resw

  @tf.function
  def opt_step(self, plan, init_feat, lam, nu):
    """ One optimization iteration """
    # TODO can use the inference distribution for init_feat rather than the point estimate
    feat_size = self._c.stoch_size + self._c.deter_size
    init_residual_func = lambda x: tf.concat(
      [(x[:, :feat_size] - init_feat),
       x[:, feat_size:-self._actdim] - tf.ones_like(init_feat) * 1e-6], -1) * 1000

    pair_residual_func = lambda x_a, x_b: self.pair_residual_func_body(x_a, x_b, lam, nu)[0]
    plan = gn_solver.solve_step(pair_residual_func, init_residual_func, plan, damping=self._c.gn_damping)
    return plan

  def _plan(self, init_obs, save_images, step, init_feat=None, verbose=True, log_extras=False):
    hor = self._c.planning_horizon
    feat_size = self._c.stoch_size + self._c.deter_size
    var_len_step = feat_size * 2 + self._actdim
    batch = self._c.n_parallel_plans
    dyn_threshold = self._c.dyn_threshold
    act_threshold = self._c.act_threshold

    if init_feat is None:
      init_feat, _ = self.get_init_feat(init_obs)
    plan = tf.random.normal((batch, (hor + 1) * var_len_step,), dtype=self._float)
    # Set the first state to be the observed initial state
    plan = tf.concat([tf.repeat(init_feat, batch, 0), plan[:, feat_size:]], 1)
    plan = tf.reshape(plan, [batch, hor + 1, var_len_step])
    # Initialize variance
    init_std = 1e-6
    plan = tf.concat([plan[..., :feat_size],
                      tf.ones_like(plan[..., feat_size:-self._actdim]) * init_std,
                      plan[..., -self._actdim:]], 2)
    lam = tf.ones((batch, hor)) * self._c.init_lam
    nu = tf.ones((batch, hor)) * self._c.init_nu

    # Run second-order solver
    plans = [plan]
    metrics = tools.AttrDefaultDict(list)
    for i in range(self._c.optimization_steps):
      plan = self.opt_step(plan, init_feat, lam, nu)
      plan_res = tf.reshape(plan, [batch, hor+1, -1])
      feat_preds, feat_var, act_preds = tf.split(plan_res, [feat_size, feat_size, self._actdim], 2)
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
      plans.append(plan)
      metrics.dynamics.append(tf.reduce_sum(dyn_viol))
      metrics.action_violation.append(tf.reduce_sum(act_viol))

      if self._c.log_colloc_scalars:
        # Compute and record dynamics loss and reward
        init_loss = tf.linalg.norm(feat_preds[:, 0] - init_feat)
        rew_raw = self._reward(feat_preds).mode()

        # Record losses and effective coefficients
        metrics.rewards.append(tf.reduce_sum(rew_raw, 1))
        metrics.dynamics_coeff.append(tf.reduce_sum(lam))
        metrics.action_coeff.append(tf.reduce_sum(nu))

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
