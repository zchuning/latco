import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd

from planning_agent import PlanningAgent
from utils import tools

class ShootingCEM(PlanningAgent):
  def _plan(self, obs, save_images, step, init_feat=None, verbose=True, log_extras=False, min_action=-1, max_action=1):
    horizon = self._c.planning_horizon
    mpc_steps = self._c.mpc_steps
    elite_size = int(self._c.cem_batch_size * self._c.cem_elite_ratio)
    var_len = self._actdim * horizon
    batch = self._c.cem_batch_size

    # Get initial states
    if init_feat is None:
      init_feat, _ = self.get_init_feat(obs)

    def eval_fitness(t):
      init_feats = tf.tile(init_feat, [batch, 1])
      actions = tf.reshape(t, [batch, horizon, -1])
      feats = self._dynamics.imagine_feat(actions, init_feats, deterministic=True)
      rewards = tf.reduce_sum(self._reward(feats).mode(), axis=1)
      return rewards, feats

    # CEM loop:
    rewards = []
    act_losses = []
    means = tf.zeros(var_len, dtype=self._float)
    stds = tf.ones(var_len, dtype=self._float)
    for i in range(self._c.optimization_steps):
      # Sample action sequences and evaluate fitness
      samples = tfd.MultivariateNormalDiag(means, stds).sample(sample_shape=[batch])
      samples = tf.clip_by_value(samples, min_action, max_action)
      fitness, feats = eval_fitness(samples)
      # Refit distribution to elite samples
      if self._c.agent == 'shooting_mppi':
        # MPPI
        weights = tf.expand_dims(tf.nn.softmax(self._c.mppi_gamma * fitness), axis=1)
        means = tf.reduce_sum(weights * samples, axis=0)
        stds = tf.sqrt(tf.reduce_sum(weights * tf.square(samples - means), axis=0))
        rewards.append(tf.reduce_sum(fitness * weights[:, 0]).numpy())
      elif self._c.agent == 'shooting_cem':
        # CEM
        _, elite_inds = tf.nn.top_k(fitness, elite_size, sorted=False)
        elite_samples = tf.gather(samples, elite_inds)
        means, vars = tf.nn.moments(elite_samples, 0)
        stds = tf.sqrt(vars + 1e-6)
        rewards.append(tf.reduce_mean(tf.gather(fitness, elite_inds)).numpy())
      # Log action violations
      means_pred = tf.reshape(means, [horizon, -1])
      act_pred = means_pred[:min(horizon, mpc_steps)]
      act_loss = tf.reduce_sum(tf.clip_by_value(tf.square(act_pred) - 1, 0, np.inf))
      act_losses.append(act_loss)

    means_pred = tf.reshape(means, [horizon, -1])
    act_pred = means_pred[:min(horizon, mpc_steps)]
    feat_pred = self._dynamics.imagine_feat(act_pred[None], init_feat, deterministic=True)
    curves = dict(rewards=rewards, action_violation=act_losses)
    if verbose:
      print("Final average reward: {0}".format(rewards[-1] / horizon))
      # Log curves
      self.logger.log_graph('losses', {f'{c[0]}/{step}': c[1] for c in curves.items()})
    if self._c.visualize:
      img_pred = self._decode(feat_pred[:min(horizon, mpc_steps)]).mode()[0]
    else:
      img_pred = None
    return act_pred, img_pred, feat_pred[0], {'metrics': tools.map_dict(lambda x: x[-1] / horizon, curves)}
