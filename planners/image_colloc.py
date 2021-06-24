import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd

from latco import LatCo, preprocess
from planners import gn_solver
from utils import tools


class ImageColloc(LatCo):
  def _plan(self, init_obs, save_images, step, init_feat=None, verbose=True, log_extras=False):
    horizon = self._c.planning_horizon
    mpc_steps = self._c.mpc_steps
    init_im = tf.cast(init_obs['image'][0], tf.float32) / 255.0 - 0.5
    feat_size = init_obs['image'][0].size
    var_len_step = self._actdim + feat_size

    # Initialize decision variables
    if init_feat is None:
      init_feat, _ = self.get_init_feat(init_obs)
    # There is one extra image at the end for convenience (because the observe(.) takes one extra action)
    t = tf.Variable(tf.random.normal((horizon + 1, var_len_step), dtype=self._float))
    lambdas = tf.ones(horizon)
    nus = tf.ones([horizon, self._actdim])
    opt = tf.keras.optimizers.Adam(learning_rate=self._c.gd_lr)

    # Gradient descent loop
    metrics = tools.AttrDefaultDict(list)
    for i in range(self._c.optimization_steps):
      # print("Gradient descent step {0}".format(i + 1))
      actions, act_viol, dyn_loss, feats_pred, grad, reward, img_pred = self.opt_step(horizon, init_im, lambdas, nus, t)
      opt.apply_gradients([(grad, t)])

      metrics.dynamics.append(tf.reduce_sum(dyn_loss))
      metrics.action_violation.append(tf.reduce_sum(act_viol))
      metrics.rewards.append(reward)
      metrics.dynamics_coeff.append(self._c.dyn_loss_scale * tf.reduce_sum(lambdas))
      metrics.action_coeff.append(self._c.act_loss_scale * tf.reduce_sum(nus))

      if self._c.log_colloc_scalars:
        # Record model rewards
        model_feats = self._dynamics.imagine_feat(actions[:, :-1], init_feat, deterministic=True)
        model_rew = self._reward(model_feats[0:1]).mode()
        metrics.model_rewards.append(tf.reduce_sum(model_rew))

      if i % self._c.lm_update_every == self._c.lm_update_every - 1:
        lambdas += self._c.lam_lr * dyn_loss
        nus += self._c.nu_lr * (act_viol)

    act_pred = t[:min(horizon, mpc_steps), :self._actdim]
    img_pred = self._decode(feats_pred[0, 1:]).mode()
    if verbose and self._c.log_colloc_scalars:
      print(f"Planned average dynamics loss: {metrics.dynamics[-1] / horizon}")
      print(f"Planned average action violation: {metrics.action_violation[-1] / horizon}")
      print(f"Planned total reward: {metrics.rewards[-1]}")
    if save_images:
      self.logger.log_graph('losses', {f'{c[0]}/{step}': c[1] for c in metrics.items()})
      self.visualize_colloc(img_pred, act_pred, init_feat, step)
    info = {'metrics': tools.map_dict(lambda x: x[-1] / horizon if len(x) > 0 else 0, dict(metrics))}
    return act_pred, img_pred, feats_pred, info

  @tf.function
  def opt_step(self, horizon, init_im, lambdas, nus, t):
    with tf.GradientTape() as g:
      g.watch(t)
      actions = tf.expand_dims(t[:, :self._actdim], 0)
      images = tf.concat([init_im[None], tf.reshape(t[:-1, self._actdim:], [-1] + list(init_im.shape))], axis=0)
      # Reward
      embed = self._encode({'image': images})
      post, prior = self._dynamics.observe(embed[None], actions, deterministic=True, trunc_bptt=self._c.imco_trunc_bptt)
      feats_post = self._dynamics.get_feat(post)
      feats_pred = self._dynamics.get_feat(prior)
      if self._c.imco_sg:
        img_pred = self._decode(tf.stop_gradient(feats_pred)).mode()
      else:
        img_pred = self._decode((feats_pred)).mode()

      reward = tf.reduce_sum(self._reward((feats_post)).mode())
      # Dynamics loss
      dynamics_loss = tf.reduce_sum(tf.square(img_pred[0, 1:] - images[1:]), axis=[1, 2, 3])
      dyn_loss = tf.reduce_sum(lambdas * dynamics_loss)
      # Action loss
      actions_viol = tf.clip_by_value(tf.square(actions[0, :-1]) - 1, 0, np.inf)
      act_loss = tf.reduce_sum(nus * actions_viol)
      fitness = - reward + self._c.dyn_loss_scale * dyn_loss + self._c.act_loss_scale * act_loss
    grad = g.gradient(fitness, t)
    return actions, actions_viol, dynamics_loss, feats_pred, grad, reward, img_pred
