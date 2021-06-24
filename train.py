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
from tensorflow_probability import distributions as tfd

tf.get_logger().setLevel('ERROR')

import base_agent
from planning_agent import define_config
from utils import wrappers, tools


def build_agent(*args):
  # Build an appropriate agent
  config = args[0]
  if config.agent == "latco":
    from latco import LatCo
    agent = LatCo(*args)
  elif config.agent == "probabilistic_latco":
    from planners.probabilistic_latco import ProbabilisticLatCo
    agent = ProbabilisticLatCo(*args)
  elif config.agent == "image_colloc":
    from planners.image_colloc import ImageColloc
    agent = ImageColloc(*args)
  elif config.agent == "dreamer":
    from planners.dreamer import Dreamer
    agent = Dreamer(*args)
  elif config.agent == 'shooting_cem':
    from planners.shooting_cem import ShootingCEM
    agent = ShootingCEM(*args)
  elif config.agent == 'shooting_mppi':
    from planners.shooting_cem import ShootingCEM
    agent = ShootingCEM(*args)
  elif config.agent == 'shooting_gn':
    from planners.shooting_gn import ShootingGN
    agent = ShootingGN(*args)
  elif config.agent == 'shooting_gd':
    from planners.shooting_gd import ShootingGD
    agent = ShootingGD(*args)
  elif config.agent == 'shooting_ilqr':
    from planners.shooting_ilqr import ShootingiLQR
    agent = ShootingiLQR(*args)
  elif config.agent == 'latco_gd':
    from planners.latco_gd import LatCoGD
    agent = LatCoGD(*args)
  elif config.agent == 'random':
    from planners.random import RandomAgent
    agent = RandomAgent(*args)
  else:
    raise NotImplementedError
  return agent


def main(config):
  datadir = base_agent.setup(config, config.logdir)
  # Create environments.
  writer = tf.summary.create_file_writer(
      str(config.logdir), max_queue=1000, flush_millis=20000)
  writer.set_as_default()
  train_envs = [wrappers.Async(lambda: base_agent.make_env(
      config, writer, 'train', datadir, store=config.train_store), config.parallel)
      for _ in range(config.envs)]
  test_envs = [wrappers.Async(lambda: base_agent.make_env(
      config, writer, 'test', datadir, store=False), config.parallel)
      for _ in range(config.envs)]
  actspace = train_envs[0].action_space

  # Prefill dataset with random episodes.
  step = base_agent.count_steps(datadir, config)
  prefill = max(0, config.prefill - step)
  print(f'Prefill dataset with {prefill} steps.')
  random_agent = lambda o, d, _: ([actspace.sample() for _ in d], None)
  tools.simulate(random_agent, train_envs, prefill / config.action_repeat)
  writer.flush()

  # Train and regularly evaluate the agent.
  step = base_agent.count_steps(datadir, config)
  print(f'Simulating agent for {config.steps-step} steps.')
  agent = build_agent(config, datadir, actspace, writer, None)
  if (config.logdir / 'variables.pkl').exists():
    print('Load checkpoint.')
    agent.load(config.logdir / 'variables.pkl')
  state = None
  while step < config.steps:
    print('Start evaluation.')
    tools.simulate(functools.partial(agent, training=False), test_envs, episodes=1)
    writer.flush()
    print('Start collection.')
    steps = config.eval_every // config.action_repeat
    state = tools.simulate(agent, train_envs, steps, state=state)
    step = base_agent.count_steps(datadir, config)
    agent.save(config.logdir / 'variables.pkl')
    if config.save_every:
      agent.save(config.logdir / f'variables_{agent.get_step() // config.save_every}.pkl')
  for env in train_envs + test_envs:
    env.close()


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
