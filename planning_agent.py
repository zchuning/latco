import argparse
import functools
import os
import pathlib
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'
sys.path.append(str(pathlib.Path(__file__).parent))

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

import base_agent
from utils import wrappers, tools


def define_config():
  config = base_agent.define_config()

  # Planning
  config.agent = 'latco'
  config.planning_horizon = 30
  config.mpc_steps = 15
  # LatCo parameters
  config.optimization_steps = 200
  config.n_parallel_plans = 1
  config.dyn_loss_scale = 1
  config.act_loss_scale = 1
  # GN parameters
  config.gn_damping = 1e-3
  config.reward_stats = True
  # Lagrange multipliers
  config.lm_update_every = 1
  config.init_lam = 1
  config.lam_lr = 1
  config.init_nu = 1
  config.nu_lr = 100
  config.dyn_threshold = 1e-4
  config.act_threshold = 1e-4
  # GD parameters
  config.gd_lr = 0.05
  # MPPI parameters
  config.mppi_gamma = 1
  # CEM parameters
  config.cem_batch_size = 10000
  config.cem_elite_ratio = 0.01
  # iLQR parameters
  config.ilqr_u_trustreg = 20
  config.ilqr_feedback_control = False
  # Probabilistic LatCo parameters
  config.n_problatco_samples = 50
  # Image collocation parameters
  config.imco_sg = False
  config.imco_trunc_bptt = True
  # Logging
  config.visualize = False
  config.logdir_eval = config.logdir  # logdir is for loading the model, logdir_eval is for output
  config.log_colloc_scalars = False
  # Eval
  config.checkpoint = 'variables.pkl'
  config.n_eval_episodes = 10
  config.store_eval_episodes = True
  return config


class PlanningAgent(base_agent.Agent):

  @tf.function
  def _policy_summaries(self, feat_pred, act_pred, init_feat):
    # Collocation
    img_pred = self._decode(feat_pred).mode()
    tools.graph_summary(self._writer, tools.video_summary, 'plan', img_pred + 0.5)

    # Forward prediction
    feat_pred = self._dynamics.imagine_feat(act_pred[None], init_feat)
    img_pred = self._decode(tf.concat((init_feat[None], feat_pred), 1)).mode()
    tools.graph_summary(self._writer, tools.video_summary, 'model', img_pred + 0.5)

    # Deterministic prediction
    feat_pred = self._dynamics.imagine_feat(act_pred[None], init_feat, deterministic=True)
    img_pred = self._decode(tf.concat((init_feat[None], feat_pred), 1)).mode()
    tools.graph_summary(self._writer, tools.video_summary, 'model_mean', img_pred + 0.5)

  def _plan(self, obs, save_images, step, init_feat=None, verbose=True, log_extras=False):
    print('This is an abstract class. The _plan function needs to be implemented')
    raise NotImplementedError

  def plan(self, feat, log_images):
    act_pred, img_pred, feat_pred, info = self._plan(None, False, None, feat, verbose=False)

    for k, v in info['metrics'].items():
      self._metrics[f'opt_{k}'].update_state(v)
    if tf.equal(log_images, True):
      self._policy_summaries(feat_pred, act_pred, feat)
    return act_pred

  def policy(self, obs, state, training, reset):
    feat, latent = self.get_init_feat(obs, state)

    if state is not None and reset.any():
      # Flush actions on reset
      state = list(state)
      state[2] = np.zeros((0,))
      state = tuple(state)

    if state is not None and state[2].shape[0] > 0:
      # Cached actions
      actions = state[2]
    else:
      actions = self.plan(feat, not training)
    action = actions[0:1]
    action = self._exploration(action, training)

    state = (latent, action, actions[1:])
    return action, state

  def forward_dynamics(self, states, actions):
    return self._dynamics.img_step(states, actions)

  def forward_dynamics_feat(self, feats, actions):
    states = self._dynamics.from_feat(feats)
    state_pred = self._dynamics.img_step(states, actions)
    feat_pred = self._dynamics.get_feat(state_pred)
    return feat_pred

  def decode_feats(self, feats):
    return self._decode(feats).mode()

  def visualize_colloc(self, img_pred, act_pred, init_feat, step=-1):
    # Use actions to predict trajectory
    feat_pred = self._dynamics.imagine_feat(act_pred[None], init_feat)
    model_imgs = self._decode(tf.concat((init_feat[None], feat_pred), 1)).mode().numpy()
    self.logger.log_video(f"model/{step}", model_imgs)

    # Deterministic prediction
    feat_pred = self._dynamics.imagine_feat(act_pred[None], init_feat, deterministic=True)
    model_imgs = self._decode(tf.concat((init_feat[None], feat_pred), 1)).mode().numpy()
    self.logger.log_video(f"model_mean/{step}", model_imgs)
