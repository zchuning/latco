import numpy as np
import tensorflow as tf
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
import tensorflow.keras.layers as layers

from planning_agent import PlanningAgent
from utils import tools


def tft(x):
  return tf.transpose(x)


class ShootingiLQR(PlanningAgent):

  def plan(self, feat, log_images):
    act_pred, img_pred, feat_pred, info = self._plan(None, None, False, None, feat, verbose=False)

    for k, v in info['metrics'].items():
      self._metrics[f'opt_{k}'].update_state(v)
    if tf.equal(log_images, True):
      self._policy_summaries(feat_pred, act_pred, feat)
    return info['controllers']

  def policy(self, obs, state, training, reset):
    feat, latent = self.get_init_feat(obs, state)

    if state is not None and reset.any():
      # Flush actions on reset
      state = list(state)
      state[2] = np.zeros((0,))
      state = tuple(state)

    if state is not None and len(state[2]) > 0:
      # Cached actions
      controllers = state[2]
    else:
      # TODO don't log every video - it takes one second
      controllers = self.plan(feat, not training)
    controller = controllers[0]
    if self._c.ilqr_feedback_control:
      action = tft(controller[0] @ (tft(feat) - controller[2]) + controller[3])
    else:
      action = tft(controller[3])
    action = self._exploration(action, training)

    state = (latent, action, controllers[1:])
    return action, state

  def _plan(self, obs, save_images, step, init_feat=None, verbose=True, log_extras=False, min_action=-1, max_action=1):
    horizon = self._c.planning_horizon
    mpc_steps = self._c.mpc_steps
    batch = self._c.n_parallel_plans
    assert batch == 1
    actdim = self._actdim

    # Get initial states
    if init_feat is None:
      init_feat, _ = self.get_init_feat(obs)

    lambdas = tf.ones([horizon, self._actdim])
    self.mu = float(self._c.ilqr_u_trustreg) # Trust region regularization
    self.alpha = 1.0 # step size
    rewards = []
    act_losses = []
    action_coeff = []
    statedim = init_feat.shape[1]
    controller_list = [(tf.zeros((actdim, statedim)), tf.zeros((actdim, 1))) for _ in range(horizon)]
    u_list = [tf.zeros((actdim, 1)) for _ in range(horizon)]
    x_list = [tf.zeros((statedim, 1)) for _ in range(horizon + 1)]
    x_list[0] = tft(init_feat)
    x_list, u_list, costs = self._forward_pass(x_list, u_list, controller_list, tf.constant(self.alpha))

    for i in range(self._c.optimization_steps):
      # Note: the final controller list is never evaluated so we don't actually know if it diverges etc.
      # Could take the one before final
      costs, x_list, u_list, controller_list = self.line_search(
        controller_list, lambdas, u_list, x_list, costs, verbose)
      rewards.append(-tf.reduce_sum(costs).numpy())
      # Log action violations
      act_pred = tf.stack(u_list[:min(horizon, mpc_steps)])[:, :, 0]
      act_loss = tf.reduce_sum(tf.clip_by_value(tf.square(act_pred) - 1, 0, np.inf))
      act_losses.append(act_loss)
      if i % self._c.lm_update_every == self._c.lm_update_every - 1:
        lambdas += self._c.lam_lr * act_loss
      action_coeff.append(tf.reduce_sum(lambdas))

    act_pred = act_pred[:min(horizon, mpc_steps)]
    feat_pred = self._dynamics.imagine_feat(act_pred[None], init_feat, deterministic=True)
    curves = dict(rewards=rewards, action_violation=act_losses, action_coeff=action_coeff)
    if verbose:
      print(f"Planned average action violation: {act_losses[-1] / horizon}")
      print("Planned total reward: {0}".format(rewards[-1]))
      # Log curves
      self.logger.log_graph('losses', {f'{c[0]}/{step}': c[1] for c in curves.items()})
    if self._c.visualize:
      img_pred = self._decode(feat_pred[:min(horizon, mpc_steps)]).mode()
    else:
      img_pred = None
    ret_controllers = [(c[0], c[1], x, u) for c, x, u in zip(controller_list, x_list, u_list)]
    return act_pred, img_pred, feat_pred, {'metrics': tools.map_dict(lambda x: x[-1] / horizon, curves),
                                           'controllers': ret_controllers}

  def line_search(self, controller_list, lambdas, u_list, x_list, costs, verbose):
    max_steps = 10
    for i in range(max_steps):
      x_old, u_old, c_old, costs_old = x_list, u_list, controller_list, costs
      costs, x_list, u_list, controller_list, quu_failure = self.opt_step(
        controller_list, lambdas, u_list, x_list, tf.constant(self.mu), tf.constant(self.alpha))

      if quu_failure:
        if verbose:
          print('quu failed')
        self.mu = self.mu * 2
        continue

      if tf.reduce_sum(costs) > tf.reduce_sum(costs_old):
        self.alpha = self.alpha / 2
        if verbose:
          print('cost not reduced', tf.reduce_sum(costs).numpy(), tf.reduce_sum(costs_old).numpy())
        x_list, u_list, controller_list, costs = x_old, u_old, c_old, costs_old
        continue
      else:
        self.alpha = min(self.alpha * 1.5, 1.0)

      break
    if i == max_steps - 1 and verbose: print('failed line search')
    return costs, x_list, u_list, controller_list

  @tf.function
  def opt_step(self, controller_list, lambdas, u_list, x_list, mu, alpha):
    """ One optimization iteration """
    quad_cost_fn = lambda x, u, lam: self.state_action_cost_quadratized(x, u, lam=lam, mu=mu)
    controller_list, quu_failure = self._backward_pass(x_list, u_list, tf.unstack(lambdas), quad_cost_fn)
    x_list, u_list, costs = self._forward_pass(x_list, u_list, controller_list, alpha)
    return costs, x_list, u_list, controller_list, quu_failure

  @tf.function
  def _forward_pass(self, x_bar_list, u_bar_list, controller_list, alpha):
    x_list = [x_bar_list[0]]
    u_list = []
    costs = []
    for i, (x_bar, u_bar, controller) in enumerate(zip(x_bar_list, u_bar_list, controller_list)):
      x = x_list[-1]
      u = controller[0] @ (x - x_bar) + alpha * controller[1] + u_bar
      # print(f"fore it: K {controller[0]}, j {controller[1]}, x {x}, x_bar {x_bar}, u_bar {u_bar}, u {u}")
      x_new = self.forward_ilqr(x_list[-1], u)

      u_list.append(u)
      x_list.append(x_new)
      cost = self.cost_ilqr(x_list[-1])
      costs.append(cost)

    return x_list, u_list, costs

  @tf.function
  def _backward_pass(self, x_list, u_list, lambdas, quad_cost_fn):
    controller_list = []
    failure_list = []

    f0, b, A = self.state_cost_quadratized(x_list[-1])
    # print(f"it0: b {b}, A {A}")
    for x, u, lam in zip(reversed(x_list[:-1]), reversed(u_list), reversed(lambdas)):
      f0, f_x, f_u = self.forward_linearized(x, u)
      K, j, A, b, failure = self.iLQR_step(A, f_x, b, f_u, u, x, lam, quad_cost_fn)
      controller_list.append((K, j))
      failure_list.append(failure)

    controller_list.reverse()
    # tf.print(failure_list)
    return controller_list, tf.reduce_any(failure_list)

  # @tf.function
  def iLQR_step(self, A, f_x, b, f_u, u, x, lam, quad_cost_fn):
    """ Propagate gains and cost-to-go backwards """
    f0, lx, lu, lxx, luu, lux = quad_cost_fn(x, u, lam=lam)

    qx = lx + tft(f_x) @ b
    qu = lu + tft(f_u) @ b
    qxx = lxx + tft(f_x) @ A @ f_x
    quu = luu + tft(f_u) @ A @ f_u

    # tf.print('quu eigs: ', tf.linalg.eigh(quu)[0])
    failure = tf.reduce_any(tf.linalg.eigh(quu)[0] < 0.01)
    # if failure:
    #   import pdb; pdb.set_trace()

    qux = lux + tft(f_u) @ A @ f_x
    # print(f"back it: b {b}, A {A}")
    # print(f"back it: quu {quu}, qu {qu}, lu {lu}, b_sys {f_u}, b {b}, luu {luu}, A {A}")

    K = -tf.linalg.solve(quu, qux)
    j = -tf.linalg.solve(quu, qu)

    A = qxx + tft(K) @ quu @ K + tft(qux) @ K + tft(K) @ qux
    b = qx + tft(K) @ quu @ j + tft(qux) @ j + tft(K) @ qu
    return K, j, A, b, failure

  def forward_ilqr(self, x, u):
    """ iLQR interface for dynamics """
    states = self._dynamics.from_feat(tft(x))
    pred = self._dynamics.img_step(states, tft(u))
    pred = self._dynamics.get_mean_feat(pred)[0]
    return pred[:, None]

  def cost_ilqr(self, x, u=None, target_state=None, target_control=None, lam=None):
    """ iLQR interface for cost """
    cost = -self._reward(tft(x)).mode()
    if u is not None:
      cost = cost + tf.reduce_sum(lam[:, None] * tf.clip_by_value(tf.square(u) - 1, 0, np.inf))
    return cost

  def forward_linearized(self, x, u):
    with tf.GradientTape(persistent=True) as tape:
      tape.watch([x, u])
      f0 = self.forward_ilqr(x, u)
    A = tape.jacobian(f0, x)[:, 0, :, 0]
    B = tape.jacobian(f0, u)[:, 0, :, 0]
    return f0, A, B

  def state_action_cost_quadratized(self, x, u, target_state=None, target_control=None, lam=None, mu=1):
    with tf.GradientTape(persistent=True) as t2:
      t2.watch([x, u])
      with tf.GradientTape(persistent=True) as t1:
        t1.watch([x, u])
        f0 = self.cost_ilqr(x, u, target_state, target_control, lam)
      lx = t1.gradient(f0, x)
      lu = t1.gradient(f0, u, unconnected_gradients=UnconnectedGradients.ZERO)
    lxx = t2.jacobian(lx, x)[:, 0, :, 0]
    lux = t2.jacobian(lu, x, unconnected_gradients=UnconnectedGradients.ZERO)[:, 0, :, 0]
    luu = t2.jacobian(lu, u, unconnected_gradients=UnconnectedGradients.ZERO)[:, 0, :, 0]
    # Trust region for actions
    luu = luu + mu * tf.eye(luu.shape[0])
    return f0, lx, lu, lxx, luu, lux

  def state_cost_quadratized(self, x, target_state=None):
    with tf.GradientTape(persistent=True) as t2:
      t2.watch(x)
      with tf.GradientTape(persistent=True) as t1:
        t1.watch(x)
        f0 = self.cost_ilqr(x, None, target_state, None)
      lx = t1.gradient(f0, x)
    lxx = t2.jacobian(lx, x)[:, 0, :, 0]
    return f0, lx, lxx
