import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd


from planning_agent import PlanningAgent
from utils import tools


class ShootingGN(PlanningAgent):
  def _plan(self, obs, save_images, step, init_feat=None, verbose=True, log_extras=False, min_action=-1, max_action=1):
    hor = self._c.planning_horizon
    mpc_steps = self._c.mpc_steps
    damping = self._c.gn_damping
    batch = self._c.n_parallel_plans
    act_threshold = self._c.act_threshold

    # Initialize decision variables
    if init_feat is None:
      init_feat, _ = self.get_init_feat(obs)
    init_feat = tf.repeat(init_feat, batch, 0)
    act_plan = tf.Variable(tf.random.normal((batch, hor*self._actdim), dtype=self._float))
    lam = tf.ones([batch, hor, self._actdim])
    act_loss, rewards = [], []

    for i in range(self._c.optimization_steps):
      J, act_viol, feats, residual, rew = self.opt_step(act_plan, batch, hor, init_feat, lam)

      # LM step
      JTJ = tf.transpose(J, [0, 2, 1]) @ J + tf.eye(hor*self._actdim, batch_shape=(batch,)) * damping
      try:
        ps_inv = tf.linalg.inv(JTJ)
      except e:
        ps_inv = tf.linalg.inv(JTJ + tf.eye(hor*self._actdim, batch_shape=(batch,)) * damping)
      dx = ps_inv @ tf.transpose(J, [0, 2, 1]) @ residual[:, :, None]
      act_plan.assign_sub(tf.squeeze(dx, 2) * self._c.gd_lr)

      # Update dual variable
      if i % self._c.lm_update_every == self._c.lm_update_every - 1:
        # lam_delta = lam * 0.1 * tf.math.log((act_viol + 0.1 * act_threshold) / act_threshold) / tf.math.log(10.0)
        # lam = lam + lam_delta
        lam += self._c.lam_lr * act_viol

      # Logging
      act_loss.append(tf.reduce_sum(act_viol))
      rewards.append(tf.reduce_sum(rew))

    rew = tf.reduce_sum(rew, 1)
    best_plan = tf.argmax(rew)
    if batch > 1:
      print(f'plan rewards: {reward}, best plan: {best_plan}')
    act_plan = tf.reshape(act_plan, [batch, hor, self._actdim])
    act_pred = act_plan[best_plan, :min(hor, mpc_steps)]
    feat_pred = feats[best_plan, :min(hor, mpc_steps)]
    curves = dict(rewards=rewards, action_violation=act_loss)
    if verbose:
      print(f"Planned average action violation: {act_loss[-1] / hor}")
      print(f"Planned total reward: {rew[best_plan]}")
      self.logger.log_graph('losses', {f'{c[0]}/{step}': c[1] for c in curves.items()})
    if self._c.visualize:
      img_pred = self._decode(feat_pred[None]).mode()
    else:
      img_pred = None
    info = {'metrics': tools.map_dict(lambda x: x[-1] / hor, curves),
            'predicted_rewards': rew[best_plan]}
    return act_pred, img_pred, feat_pred, info

  @tf.function
  def opt_step(self, act_plan, batch, hor, init_feat, lam):
    with tf.GradientTape(persistent=True) as g:
      g.watch(act_plan)
      act_plan_r = tf.reshape(act_plan, [batch, hor, self._actdim])
      feats = self._dynamics.imagine_feat(act_plan_r, init_feat, deterministic=True)
      rew = self._reward(feats).mode()
      rew_res = tf.math.softplus(-rew)
      act_viol = tf.clip_by_value(tf.abs(act_plan_r) - 1, 0, np.inf)
      act_res = tf.reduce_sum(tf.sqrt(lam) * act_viol, axis=2)
      residual = rew_res + act_res
    J = g.batch_jacobian(residual, act_plan)
    return J, act_viol, feats, residual, rew
