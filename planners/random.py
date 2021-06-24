from latco import LatCo
import tensorflow as tf
import numpy as np

class RandomAgent(LatCo):
  def _plan(self, init_obs, save_images, step, init_feat=None, verbose=True, log_extras=False):
    act_pred = tf.random.uniform((self._c.mpc_steps,) + self._actspace.shape, self._actspace.low[0], self._actspace.high[0])
    return act_pred, None, None, None
