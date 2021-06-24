import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd

from latco import LatCo, preprocess
from planners import gn_solver


class LatCoGD(LatCo):
  def collocation_gd(self, obs, save_images, step):
    horizon = self._c.planning_horizon
    mpc_steps = self._c.mpc_steps
    feat_size = self._c.stoch_size + self._c.deter_size
    var_len_step = self._actdim + feat_size

    # Initialize decision variables
    init_feat, _ = self.get_init_feat(obs)
    t = tf.Variable(tf.random.normal((horizon, var_len_step), dtype=self._float))
    lambdas = tf.ones(horizon)
    nus = tf.ones([horizon, self._actdim])
    dyn_loss, act_loss, rewards = [], [], []
    dyn_loss_frame = []
    opt = tf.keras.optimizers.Adam(learning_rate=self._c.gd_lr)
    # Gradient descent loop
    for i in range(self._c.optimization_steps):
      # print("Gradient descent step {0}".format(i + 1))
      with tf.GradientTape() as g:
        g.watch(t)
        actions = tf.expand_dims(t[:, :self._actdim], 0)
        feats = tf.concat([init_feat, t[:, self._actdim:]], axis=0)
        # Compute reward
        reward = tf.reduce_sum(self._reward(feats).mode())
        # Compute dynamics loss
        states = self._dynamics.from_feat(feats[None, :-1])
        priors = self._dynamics.img_step(states, actions)
        feats_pred = tf.squeeze(tf.concat([priors['mean'], priors['deter']], axis=-1))
        log_prob_frame = tf.reduce_sum(tf.square(feats_pred - feats[1:]), axis=1)
        log_prob = tf.reduce_sum(lambdas * log_prob_frame)
        actions_viol = tf.clip_by_value(tf.square(actions) - 1, 0, np.inf)
        actions_constr = tf.reduce_sum(nus * actions_viol)
        fitness = - reward + self._c.dyn_loss_scale * log_prob + self._c.act_loss_scale * actions_constr
      # print(f"Frame-wise log probability: {log_prob_frame}")
      # print(f"Reward: {reward}, dynamics: {log_prob}")
      grad = g.gradient(fitness, t)
      opt.apply_gradients([(grad, t)])
      dyn_loss.append(tf.reduce_sum(log_prob_frame))
      dyn_loss_frame.append(log_prob_frame)
      act_loss.append(tf.reduce_sum(actions_viol))
      rewards.append(reward)
      if i % self._c.lm_update_every == self._c.lm_update_every - 1:
        lambdas += self._c.lam_lr * log_prob_frame
        nus += self._c.nu_lr * (actions_viol)
        # print(tf.reduce_mean(log_prob_frame, axis=0))
        # print(f"Lambdas: {lambdas}\n Nus: {nus}")

    act_pred = t[:min(horizon, mpc_steps), :self._actdim]
    feat_pred = t[:min(horizon, mpc_steps), self._actdim:]
    img_pred = self._decode(feat_pred).mode()
    print(f"Final average dynamics loss: {dyn_loss[-1] / horizon}")
    print(f"Final average action violation: {act_loss[-1] / horizon}")
    print(f"Final average reward: {rewards[-1] / horizon}")
    if save_images:
      self.logger.log_graph(
        'losses', {f'rewards/{step}': rewards, f'dynamics/{step}': dyn_loss, f'action_violation/{step}': act_loss})
      self.visualize_colloc(img_pred, act_pred, init_feat)
    return act_pred, img_pred, feat_pred
