import argparse
import collections
import functools
import json
import os
import pathlib
import sys
import time
import yaml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'
sys.path.append(str(pathlib.Path(__file__).parent))

import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import distributions as tfd

tf.get_logger().setLevel('ERROR')

from utils import wrappers, tools, models


def define_config():
  config = tools.AttrDict()
  # General.
  config.logdir = pathlib.Path('.')
  config.offline_dir = pathlib.Path('.')
  config.seed = 0
  config.steps = 5e5
  config.eval_every = 1e4
  config.log_every = 1e3
  config.log_scalars = True
  config.log_images = True
  config.gpu_growth = True
  config.precision = 32
  config.additional_gifs = False
  # Environment.
  config.task = 'mw_SawyerReachEnvV2'
  config.envs = 1
  config.parallel = 'none'
  config.action_repeat = 1
  config.time_limit = 150
  config.prefill = 5000
  config.eval_noise = 0.0
  config.clip_rewards = 'none'
  config.collect_sparse_reward = False
  config.use_sparse_reward = False
  config.state_size = 9
  # Model.
  config.deter_size = 200
  config.stoch_size = 30
  config.num_units = 400
  config.dense_act = 'elu'
  config.cnn_act = 'relu'
  config.cnn_depth = 32
  config.pcont = False
  config.free_nats = 3.0
  config.kl_scale = 1.0
  config.pcont_scale = 10.0
  config.weight_decay = 0.0
  config.weight_decay_pattern = r'.*'
  config.state_regressor = False
  config.save_every = 100000
  config.normalize_reward = False
  # Training.
  config.batch_size = 50
  config.batch_length = 50
  config.train_every = 150
  config.train_steps = 15
  config.train_store = True
  config.pretrain = 100
  config.model_lr = 6e-4
  config.value_lr = 8e-5
  config.actor_lr = 8e-5
  config.grad_clip = 100.0
  config.dataset_balance = False
  # Behavior.
  config.discount = 0.99
  config.disclam = 0.95
  config.horizon = 15
  config.action_dist = 'tanh_normal'
  config.action_init_std = 5.0
  config.expl = 'additive_gaussian'
  config.expl_amount = 0.3
  config.expl_decay = 0.0
  config.expl_min = 0.0
  return config


class Agent(tools.Module):

  def __init__(self, config, datadir, actspace, writer=None, logger=None):
    self._c = config
    self._actspace = actspace
    self._actdim = actspace.n if hasattr(actspace, 'n') else actspace.shape[0]
    self._writer = writer
    self.logger = logger
    self._random = np.random.RandomState(config.seed)
    with tf.device('cpu:0'):
      self._step = tf.Variable(count_steps(datadir, config), dtype=tf.int64)
    self._should_pretrain = tools.Once()
    self._should_train = tools.Every(config.train_every)
    self._should_log = tools.Every(config.log_every)
    self._last_log = None
    self._last_time = time.time()
    self._metrics = collections.defaultdict(tf.metrics.Mean)
    # Create variables for checkpoint
    self._metrics['expl_amount']
    self._metrics['opt_rewards']
    self._metrics['opt_dynamics']
    self._metrics['opt_action_violation']
    self._metrics['opt_dynamics_coeff']
    self._metrics['opt_action_coeff']
    self._metrics['opt_model_rewards']
    self._float = prec.global_policy().compute_dtype
    self._dataset = iter(load_dataset(datadir, self._c))
    self._build_model()
    # Do a train step to initialize all variables, including optimizer
    # statistics. Ideally, we would use batch size zero, but that doesn't work
    # in multi-GPU mode.
    self.train(next(self._dataset))

  def __call__(self, obs, reset, state=None, training=True):
    step = self._step.numpy().item()
    tf.summary.experimental.set_step(step)
    if state is not None and reset.any():
      mask = tf.cast(1 - reset, self._float)[:, None]
      state = tf.nest.map_structure(lambda x: x * mask, state)
    if training and self._should_train(step):
      log = self._should_log(step)
      n = self._c.pretrain if self._should_pretrain() else self._c.train_steps
      print(f'Training for {n} steps.')
      for train_step in range(n):
        log_images = self._c.log_images and log and train_step == 0
        self.train(next(self._dataset), log_images)
      if log:
        self._write_summaries()
    action, state = self.policy(obs, state, training, reset)
    if training:
      self._step.assign_add(len(reset) * self._c.action_repeat)
    return action, state

  @tf.function
  def policy(self, obs, state, training, reset):
    feat, latent = self.get_init_feat(obs, state)
    if training:
      action = self._actor(feat).sample()
    else:
      action = self._actor(feat).mode()
    action = self._exploration(action, training)
    state = (latent, action)
    return action, state

  def get_init_feat(self, obs, state=None):
    if state is None:
      latent = self._dynamics.initial(len(obs['image']))
      action = tf.zeros((len(obs['image']), self._actdim), self._float)
    else:
      latent, action = state[:2]
    embed = self._encode(preprocess(obs, self._c))
    latent, _ = self._dynamics.obs_step(latent, action, embed)
    feat = self._dynamics.get_feat(latent)
    return feat, latent

  def load(self, filename):
    super().load(filename)
    self._should_pretrain()

  @tf.function
  def train(self, data, log_images=False):
    with tf.GradientTape() as model_tape:
      embed = self._encode(data)
      post, prior = self._dynamics.observe(embed, data['action'])
      feat = self._dynamics.get_feat(post)
      image_pred = self._decode(feat)
      reward_pred = self._reward(feat)
      likes = tools.AttrDict()
      likes.image = tf.reduce_mean(image_pred.log_prob(data['image']))
      if self._c.use_sparse_reward:
        reward = reward_unnorm = data['sparse_reward']
      else:
        reward = reward_unnorm = data['reward']
      if self._c.normalize_reward:
        reward = (reward - self.r_roll_mean) / self.r_roll_std
      likes.reward = tf.reduce_mean(reward_pred.log_prob(reward))
      if self._c.state_regressor:
        states_pred = self._state(tf.stop_gradient(feat))
        likes.state_regressor = tf.reduce_mean(states_pred.log_prob(data['state']))
      if self._c.pcont:
        pcont_pred = self._pcont(feat)
        pcont_target = self._c.discount * data['discount']
        likes.pcont = tf.reduce_mean(pcont_pred.log_prob(pcont_target))
        likes.pcont *= self._c.pcont_scale
      prior_dist = self._dynamics.get_dist(prior)
      post_dist = self._dynamics.get_dist(post)
      div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist))
      div = tf.maximum(div, self._c.free_nats)
      model_loss = self._c.kl_scale * div - sum(likes.values())

    if self._c.reward_stats:
      roll = 0.99
      mean = tf.reduce_mean(reward_unnorm, [0, 1])
      std = tf.math.reduce_std(reward_unnorm, [0, 1])
      self.r_roll_mean.assign(mean * (1 - roll) + self.r_roll_mean * roll)
      self.r_roll_std.assign(std * (1 - roll) + self.r_roll_std * roll)

    model_norm = self._model_opt(model_tape, model_loss)
    self._reward.train(feat, reward, reward_pred)

    if self._c.log_scalars:
      self._scalar_summaries(
          data, feat, prior_dist, post_dist, likes, div,
          model_loss, model_norm)
    if tf.equal(log_images, True):
      self._image_summaries(data, embed, image_pred)

    return post, feat

  def _build_model(self):
    acts = dict(
        elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
        leaky_relu=tf.nn.leaky_relu)
    cnn_act = acts[self._c.cnn_act]
    act = acts[self._c.dense_act]
    self._encode = models.ConvEncoder(self._c.cnn_depth, cnn_act)
    if self._c.deter_size > 0:
      self._dynamics = models.RSSM(self._c.stoch_size, self._c.deter_size, self._c.deter_size)
    else:
      self._dynamics = models.SSM(self._c.stoch_size, 2, self._c.num_units)
    self._decode = models.ConvDecoder(self._c.cnn_depth, cnn_act)
    self._reward = models.DenseDecoder((), 2, self._c.num_units, act=act)
    if self._c.reward_stats:
      self.r_roll_mean = tf.Variable(tf.zeros((1,)))
      self.r_roll_std = tf.Variable(tf.ones((1,)))
    if self._c.state_regressor:
      self._state = models.DenseDecoder((self._c.state_size,), 2, self._c.num_units, act=act)
    if self._c.pcont:
      self._pcont = models.DenseDecoder(
          (), 3, self._c.num_units, 'binary', act=act)
    model_modules = [self._encode, self._dynamics, self._decode, self._reward]
    if self._c.state_regressor:
      model_modules.append(self._state)
    if self._c.pcont:
      model_modules.append(self._pcont)
    Optimizer = functools.partial(
        tools.Adam, wd=self._c.weight_decay, clip=self._c.grad_clip,
        wdpattern=self._c.weight_decay_pattern)
    self._model_opt = Optimizer('model', model_modules, self._c.model_lr)

  def _exploration(self, action, training):
    if training:
      amount = self._c.expl_amount
      if self._c.expl_decay:
        amount *= 0.5 ** (tf.cast(self._step, tf.float32) / self._c.expl_decay)
      if self._c.expl_min:
        amount = tf.maximum(self._c.expl_min, amount)
      self._metrics['expl_amount'].update_state(amount)
    elif self._c.eval_noise:
      amount = self._c.eval_noise
    else:
      return action
    if self._c.expl == 'additive_gaussian':
      return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
    if self._c.expl == 'completely_random':
      return tf.random.uniform(action.shape, -1, 1)
    if self._c.expl == 'epsilon_greedy':
      indices = tfd.Categorical(0 * action).sample()
      return tf.where(
          tf.random.uniform(action.shape[:1], 0, 1) < amount,
          tf.one_hot(indices, action.shape[-1], dtype=self._float),
          action)
    raise NotImplementedError(self._c.expl)

  def _imagine_ahead(self, post):
    if self._c.pcont:  # Last step could be terminal.
      post = {k: v[:, :-1] for k, v in post.items()}
    flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in post.items()}
    policy = lambda state: self._actor(
        tf.stop_gradient(self._dynamics.get_feat(state))).sample()
    states = tools.static_scan(
        lambda prev, _: self._dynamics.img_step(prev, policy(prev)),
        tf.range(self._c.horizon), start)
    imag_feat = self._dynamics.get_feat(states)
    return imag_feat

  def _scalar_summaries(
      self, data, feat, prior_dist, post_dist, likes, div,
      model_loss, model_norm):
    self._metrics['model_grad_norm'].update_state(model_norm)
    self._metrics['prior_ent'].update_state(prior_dist.entropy())
    self._metrics['post_ent'].update_state(post_dist.entropy())
    for name, logprob in likes.items():
      self._metrics[name + '_loss'].update_state(-logprob)
    self._metrics['div'].update_state(div)
    self._metrics['model_loss'].update_state(model_loss)
    if self._c.reward_stats:
      self._metrics['r_roll_mean'].update_state(self.r_roll_mean)
      self._metrics['r_roll_std'].update_state(self.r_roll_std)

  def _image_summaries(self, data, embed, image_pred):
    n_cond = 5
    truth = data['image'][:6] + 0.5
    recon = image_pred.mode()[:6]
    init, _ = self._dynamics.observe(embed[:6, :n_cond], data['action'][:6, :n_cond])
    init = {k: v[:, -1] for k, v in init.items()}
    prior = self._dynamics.imagine(data['action'][:6, n_cond:], init)
    openl = self._decode(self._dynamics.get_feat(prior)).mode()
    openl = tf.concat([recon[:, :n_cond], openl], 1)
    self.log_video(openl, truth, 'agent/openl')
    del openl

    if self._c.additional_gifs:
      self.log_video(recon, truth, 'inference')
      post, prior = self._dynamics.observe(embed[:6], data['action'][:6])
      closedl = self._decode(self._dynamics.get_feat(prior)).mode()
      closedl = tf.concat([recon[:, :1], closedl[:, 1:]], 1)
      self.log_video(closedl, truth, 'closedl')

  def log_video(self, pred, truth, name):
    model = pred + 0.5
    error = (model - truth + 1) / 2
    pred = tf.concat([truth, model, error], 2)
    tools.graph_summary(self._writer, tools.video_summary, name, pred)

  def _write_summaries(self):
    step = int(self._step.numpy())
    metrics = [(k, float(v.result())) for k, v in self._metrics.items()]
    if self._last_log is not None:
      duration = time.time() - self._last_time
      self._last_time += duration
      metrics.append(('fps', (step - self._last_log) / duration))
    self._last_log = step
    [m.reset_states() for m in self._metrics.values()]
    with (self._c.logdir / 'metrics.jsonl').open('a') as f:
      f.write(json.dumps({'step': step, **dict(metrics)}) + '\n')
    [tf.summary.scalar('agent/' + k, m) for k, m in metrics]
    print(f'[{step}]: {self._c.logdir} , ', ' / '.join(f'{k} {v:.1f}' for k, v in metrics))
    self._writer.flush()

  def get_step(self):
    return int(self._step.numpy())


def preprocess(obs, config):
  dtype = prec.global_policy().compute_dtype
  obs = obs.copy()
  with tf.device('cpu:0'):
    obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5

    if config.clip_rewards[0] == 'd':
      clip_rewards = lambda r: r / float(config.clip_rewards[1:])
    else:
      clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[config.clip_rewards]
    if 'reward' in obs:
      obs['reward'] = clip_rewards(obs['reward'])
    for k, v in obs.items():
      obs[k] = tf.cast(v, dtype)
  return obs


def count_steps(datadir, config):
  return tools.count_episodes(datadir)[1] * config.action_repeat


def load_dataset(directory, config):
  episode = next(tools.load_episodes(directory, 1, offline_dir=config.offline_dir))
  types = {k: v.dtype for k, v in episode.items()}
  shapes = {k: (None,) + v.shape[1:] for k, v in episode.items()}
  generator = lambda: tools.load_episodes(
      directory, config.train_steps, config.batch_length,
      config.dataset_balance, offline_dir=config.offline_dir)
  dataset = tf.data.Dataset.from_generator(generator, types, shapes)
  dataset = dataset.batch(config.batch_size, drop_remainder=True)
  dataset = dataset.map(functools.partial(preprocess, config=config))
  dataset = dataset.prefetch(10)
  return dataset


def summarize_episode(episode, config, datadir, writer, prefix):
  episodes, _ = tools.count_episodes(datadir)
  length = (len(episode['reward']) - 1) * config.action_repeat
  ret = episode['reward'].sum()
  print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
  metrics = [
      (f'{prefix}/return', float(episode['reward'].sum())),
      (f'{prefix}/length', len(episode['reward']) - 1),
      (f'episodes', episodes)]
  if config.collect_sparse_reward:
    metrics.append((f'{prefix}/sparse_return', float(episode['sparse_reward'].sum())))
    metrics.append((f'{prefix}/success', float(episode['sparse_reward'].sum() > 0)))

  for key in filter(lambda k: 'metric_' in k, episode):
    metric_min = np.min(episode[key].astype(np.float64))
    metric_max = np.max(episode[key].astype(np.float64))
    metric_mean = float(episode[key].astype(np.float64).mean())
    metric_final = float(episode[key].astype(np.float64)[-1])
    key = key.replace('metric_', '')

    metrics.append((f'{prefix}/min_{key}', metric_min))
    metrics.append((f'{prefix}/max_{key}', metric_max))
    metrics.append((f'{prefix}/mean_{key}', metric_mean))
    metrics.append((f'{prefix}/final_{key}', metric_final))

  step = count_steps(datadir, config)
  with (config.logdir / 'metrics.jsonl').open('a') as f:
    f.write(json.dumps(dict([('step', step)] + metrics)) + '\n')
  with writer.as_default():  # Env might run in a different thread.
    tf.summary.experimental.set_step(step)
    [tf.summary.scalar('sim/' + k, v) for k, v in metrics]

    if prefix == 'test':
      tools.video_summary(f'sim/{prefix}/video', episode['image'][None], step)


def make_env(config, writer, prefix, datadir, store):
  env = make_bare_env(config)
  env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
  callbacks = []
  if store:
    callbacks.append(lambda ep: tools.save_episodes(datadir, [ep]))
  callbacks.append(
      lambda ep: summarize_episode(ep, config, datadir, writer, prefix))
  env = wrappers.Collect(env, callbacks, config.precision, config.collect_sparse_reward)
  env = wrappers.RewardObs(env)
  return env


def make_bare_env(config):
  suite, task = config.task.split('_', 1)
  if suite == 'dmc':
    env = wrappers.DeepMindControl(task)
    env = wrappers.ActionRepeat(env, config.action_repeat)
    env = wrappers.NormalizeActions(env)
  elif suite == "mw":
    import envs.sparse_metaworld
    env = envs.sparse_metaworld.SparseMetaWorld(task, config.action_repeat)
  elif suite == "pointmass":
    env = wrappers.PointmassEnv(task, config.action_repeat)
  else:
    raise NotImplementedError(suite)
  return env


def setup(config, logdir):
  if config.gpu_growth:
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    prec.set_policy(prec.Policy('mixed_float16'))
  config.steps = int(config.steps)
  logdir.mkdir(parents=True, exist_ok=True)
  print('Logdir', logdir)
  datadir = logdir / 'episodes'

  # Save run parameters
  tools.save_cmd(logdir)
  tools.save_git(logdir)
  with open(logdir / 'config.yaml', 'w') as yaml_file: yaml.dump(config, yaml_file, default_flow_style=False)
  return datadir


def main(config):
  datadir = setup(config, config.logdir)
  # Create environments.
  writer = tf.summary.create_file_writer(
    str(config.logdir), max_queue=1000, flush_millis=20000)
  writer.set_as_default()
  train_envs = [wrappers.Async(lambda: make_env(
      config, writer, 'train', datadir, store=config.train_store), config.parallel)
      for _ in range(config.envs)]
  test_envs = [wrappers.Async(lambda: make_env(
      config, writer, 'test', datadir, store=False), config.parallel)
      for _ in range(config.envs)]
  actspace = train_envs[0].action_space

  # Prefill dataset with random episodes.
  step = count_steps(datadir, config)
  prefill = max(0, config.prefill - step)
  print(f'Prefill dataset with {prefill} steps.')
  random_agent = lambda o, d, _: ([actspace.sample() for _ in d], None)
  tools.simulate(random_agent, train_envs, prefill / config.action_repeat)
  writer.flush()

  # Train and regularly evaluate the agent.
  step = count_steps(datadir, config)
  print(f'Simulating agent for {config.steps-step} steps.')
  agent = Agent(config, datadir, actspace, writer)
  if (config.logdir / 'variables.pkl').exists():
    print('Load checkpoint.')
    agent.load(config.logdir / 'variables.pkl')
  state = None
  while step < config.steps:
    print('Start evaluation.')
    tools.simulate(
        functools.partial(agent, training=False), test_envs, episodes=1)
    writer.flush()
    print('Start collection.')
    steps = config.eval_every // config.action_repeat
    state = tools.simulate(agent, train_envs, steps, state=state)
    step = count_steps(datadir, config)
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
