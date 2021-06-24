import argparse
import os
import pathlib
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'
sys.path.append(str(pathlib.Path(__file__).parent))

import imageio
import numpy as np
import pickle
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp

tf.get_logger().setLevel('ERROR')

import base_agent
import planning_agent
from train import build_agent
from base_agent import Agent, preprocess, make_bare_env
from planning_agent import define_config
from planners import gn_solver
from utils import logging, wrappers, tools


def make_env(config):
  env = make_bare_env(config)
  env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
  if config.store_eval_episodes:
    datadir = config.logdir_eval / 'eval_episodes'
    callbacks = [lambda ep: tools.save_episodes(datadir, [ep])]
    env = wrappers.Collect(env, callbacks, config.precision, config.collect_sparse_reward)
  env = wrappers.RewardObs(env)
  return env


def colloc_simulate(agent, config, env, save_images=True):
  """ Run planning loop """
  # Define task-related variables
  obs = env.reset()
  obs['image'] = [obs['image']]
  ep_length = config.time_limit // config.action_repeat

  # Simulate one episode
  img_preds, act_preds, frames = [], [], []
  total_reward, total_sparse_reward, total_predicted_reward = 0, 0, 0
  start = time.time()
  for t in range(0, ep_length, config.mpc_steps):
    # Run single planning step
    print("Planning step {0} of {1}".format(t + 1, ep_length))
    act_pred, img_pred, feat_pred, info = agent._plan(obs, save_images, t)

    # Accumulate predicted reward
    if info is not None:
      if 'predicted_rewards' in info:
        total_predicted_reward += info['metrics']['predicted_rewards']
      elif 'metrics' in info:
        total_predicted_reward += info['metrics']['rewards'] if 'rewards' in info['metrics'] else 0

    # Simluate in environment
    act_pred_np = act_pred.numpy()
    for i in range(min(len(act_pred_np), ep_length - t)):
      obs, reward, done, info = env.step(act_pred_np[i])
      total_reward += reward
      if 'success' in info:
        total_sparse_reward += info['success'] # float(info['goalDist'] < 0.15)
      frames.append(obs['image'])
    obs['image'] = [obs['image']]

    # Logging
    act_preds.append(act_pred_np)
    if img_pred is not None:
      img_preds.append(img_pred.numpy())
      agent.logger.log_video(f"plan/{t}", img_pred.numpy())
    agent.logger.log_video(f"execution/{t}", frames[-len(act_pred_np):])
  end = time.time()
  print(f"Episode time: {end - start}")
  print(f"Total predicted reward: {total_predicted_reward}")
  print(f"Total reward: {total_reward}")
  agent.logger.log_graph('predicted_reward', {'rewards/predicted':[total_predicted_reward]})
  agent.logger.log_graph('true_reward', {'rewards/true': [total_reward]})

  if 'success' in info:
    success = float(total_sparse_reward > 0) # info['success']
    print(f"Total sparse reward: {total_sparse_reward}")
    agent.logger.log_graph('true_sparse_reward', {'rewards/true': [total_sparse_reward]})
    print(f"Success: {success}")
  else:
    success = np.nan
  if 'goalDist' in info and info['goalDist'] is not None:
    goal_dist = info['goalDist']
  elif 'reachDist' in info:
    goal_dist = info['reachDist']
  else:
    goal_dist = np.nan
  if save_images:
    if img_pred is not None:
      img_preds = np.vstack(img_preds)
      agent.logger.log_video("plan/full", img_preds)
    agent.logger.log_video("execution/full", frames)

  ep_info = dict(reward_dense=total_reward,
                 reward_pred=total_predicted_reward,
                 success=success,
                 goal_dist=goal_dist)
  if config.collect_sparse_reward:
    ep_info['reward_sparse'] = total_sparse_reward
  return ep_info


def main(config):
  base_agent.setup(config, config.logdir_eval)
  env = make_env(config)
  logger = logging.TBLogger(config.logdir_eval)
  datadir = config.logdir / 'episodes'
  actspace = env.action_space
  agent = build_agent(config, datadir, actspace, None, logger)
  agent.load(config.logdir / config.checkpoint)
  tf.summary.experimental.set_step(0)

  run_metrics = tools.AttrDefaultDict(list)
  for i in range(config.n_eval_episodes):
    print(f'------- Evaluating plan {i} of {config.n_eval_episodes}')
    ep_info = colloc_simulate(agent, config, env, True)
    for k, v in ep_info.items():
      run_metrics[k].append(v)

  print(f'------- Finished {config.n_eval_episodes}')
  for k, v in run_metrics.items():
    save_key = 'total_reward/' + k.replace('reward_', '')
    print(f'Average {k}: {np.mean(v)}')
    agent.logger.log_graph(None, {save_key + '_std': [np.std(v)]})
    agent.logger.log_graph(None, {save_key + '_mean': [np.mean(v)]})

  key = 'reward_sparse' if config.use_sparse_reward else 'reward_dense'
  agent.logger.log_scatter('obtained_predicted_reward', np.stack([run_metrics[key], run_metrics.reward_pred], 0))
  with (config.logdir_eval / 'eval_data.pkl').open('wb') as f:
    pickle.dump(dict(run_metrics), f)


if __name__ == '__main__':
  try:
    import colored_traceback
    colored_traceback.add_hook()
  except ImportError:
    pass
  parser = argparse.ArgumentParser()
  for key, value in define_config().items():
    parser.add_argument(f'--{key}', type=tools.args_type(value), default=value)
  main(parser.parse_args())
