import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd


from planning_agent import PlanningAgent
from utils import tools


class ShootingGD(PlanningAgent):
  def _plan(self, obs, save_images, step, init_feat=None, verbose=True, log_extras=False, min_action=-1, max_action=1):
    horizon = self._c.planning_horizon
    mpc_steps = self._c.mpc_steps
    batch = self._c.n_parallel_plans

    # Initialize decision variables
    if init_feat is None:
      init_feat, _ = self.get_init_feat(obs)
    init_feat = tf.repeat(init_feat, batch, 0)
    action_plan = tf.Variable(tf.random.normal((batch, horizon, self._actdim), dtype=self._float))
    lambdas = tf.ones([batch, horizon, self._actdim])
    act_loss, rewards = [], []
    opt = tf.keras.optimizers.Adam(learning_rate=self._c.gd_lr)
    # Gradient descent loop
    for i in range(self._c.optimization_steps):
      # print("Gradient descent step {0}".format(i + 1))
      with tf.GradientTape() as g:
        g.watch(action_plan)
        feats = self._dynamics.imagine_feat(action_plan, init_feat, deterministic=True)
        reward = tf.reduce_sum(self._reward(feats).mode(), 1)
        actions_viol = tf.clip_by_value(tf.square(action_plan) - 1, 0, np.inf)
        actions_constr = tf.reduce_sum(lambdas * actions_viol)
        fitness = - tf.reduce_sum(reward) + self._c.act_loss_scale * actions_constr
      grad = g.gradient(fitness, action_plan)
      opt.apply_gradients([(grad, action_plan)])
      action_plan.assign(tf.clip_by_value(action_plan, min_action, max_action)) # Prevent OOD preds
      act_loss.append(tf.reduce_sum(actions_viol))
      rewards.append(tf.reduce_sum(reward))
      if i % self._c.lm_update_every == self._c.lm_update_every - 1:
        lambdas += self._c.lam_lr * actions_viol

    best_plan = tf.argmax(reward)
    if batch > 1:
      print(f'plan rewards: {reward}, best plan: {best_plan}')
    act_pred = action_plan[best_plan, :min(horizon, mpc_steps)]
    feat_pred = feats
    curves = dict(rewards=rewards, action_violation=act_loss)
    if verbose:
      print(f"Planned average action violation: {act_loss[-1] / horizon}")
      print(f"Planned total reward: {reward[best_plan]}")
      self.logger.log_graph('losses', {f'{c[0]}/{step}': c[1] for c in curves.items()})
    if self._c.visualize:
      img_pred = self._decode(feat_pred[best_plan, :min(horizon, mpc_steps)][None]).mode()[0]
    else:
      img_pred = None
    return act_pred, img_pred, feat_pred, {'metrics': tools.map_dict(lambda x: x[-1] / horizon, curves),
                                           'predicted_rewards': reward[best_plan]}
