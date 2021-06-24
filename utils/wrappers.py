import atexit
import functools
import sys
import threading
import traceback
import os
import re

import gym
import numpy as np
from PIL import Image

if 'MUJOCO_RENDERER' in os.environ:
  RENDERER = os.environ['MUJOCO_RENDERER']
else:
  RENDERER = 'glfw'


class DreamerEnv():

  LOCK = threading.Lock()

  def __init__(self, action_repeat, width=64):
    self._action_repeat = action_repeat
    self._width = width
    self._size = (self._width, self._width)

  @property
  def observation_space(self):
    shape = self._size + (3,)
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return self._env.action_space

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      state = self._env.reset()
    return self._get_obs(state)

  def step(self, action):
    total_reward = 0.0
    for step in range(self._action_repeat):
      state, reward, done, info = self._env.step(action)
      total_reward += reward
      if done:
        break
    obs = self._get_obs(state)
    return obs, total_reward, done, info

  def render(self, mode):
    return self._env.render(mode)

  def _get_obs(self, state):
    self._offscreen.render(self._width, self._width, -1)
    image = np.flip(self._offscreen.read_pixels(self._width, self._width)[0], 1)

    obs = {'image': image}
    if isinstance(state, dict):
      obs.update(state)
    else:
      obs['state'] = state
    return obs


class PointmassEnv(DreamerEnv):

  def __init__(self, task=None, action_repeat=1):
    super().__init__(action_repeat)
    from mujoco_py import MjRenderContext
    from envs.pointmass.pointmass_prob_env import PointmassProb
    # from envs.pointmass.pointmass_smart_env import Pointmass as PointmassSmart
    with self.LOCK:
      task = 'pm_probabilistic_20pc_10nrew_2prew'
      self._task_keys, task = parse(task, ['(\d*)pc', '(\d*)nrew', '(\d*)prew'])
      self._env = PointmassProb(int(self._task_keys['Npc']) / 100,
                                int(self._task_keys['Nnrew']),
                                int(self._task_keys['Nprew']))

    self._offscreen = MjRenderContext(self._env.sim, True, 0, RENDERER, True)
    set_camera(self._offscreen.cam, azimuth=0, elevation=90, distance=2.6)


class DeepMindControl:

  def __init__(self, name, size=(64, 64), camera=None):
    domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if isinstance(domain, str):
      from dm_control import suite
      self._env = suite.load(domain, task)
    else:
      assert task is None
      self._env = domain()
    self._size = size
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera = camera

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

  def step(self, action):
    time_step = self._env.step(action)
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    reward = time_step.reward or 0
    done = time_step.last()
    info = {'discount': np.array(time_step.discount, np.float32)}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.physics.render(*self._size, camera_id=self._camera)


class Collect:

  def __init__(self, env, callbacks=None, precision=32, save_sparse_reward=False):
    self._env = env
    self._callbacks = callbacks or ()
    self._precision = precision
    self._episode = None
    self._save_sparse_reward = save_sparse_reward

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    obs_or, reward, done, info = self._env.step(action)
    obs = {k: self._convert(v) for k, v in obs_or.items()}
    transition = obs.copy()
    transition['action'] = action
    transition['reward'] = reward
    if self._save_sparse_reward:
      transition['sparse_reward'] = info.get('success')
    transition['discount'] = info.get('discount', np.array(1 - float(done)))
    self._episode.append(transition)
    if done:
      episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
      episode = {k: self._convert(v) for k, v in episode.items()}
      info['episode'] = episode
      for callback in self._callbacks:
        callback(episode)
    return obs_or, reward, done, info

  def reset(self):
    obs = self._env.reset()
    transition = obs.copy()
    transition['action'] = np.zeros(self._env.action_space.shape)
    transition['reward'] = 0.0
    transition['discount'] = 1.0
    if self._save_sparse_reward:
      transition['sparse_reward'] = 0.0
    self._episode = [transition]
    return obs

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      dtype = {16: np.float16, 32: np.float32, 64: np.float64}[self._precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
      dtype = {16: np.int16, 32: np.int32, 64: np.int64}[self._precision]
    elif np.issubdtype(value.dtype, np.uint8):
      dtype = np.uint8
    else:
      raise NotImplementedError(value.dtype)
    return value.astype(dtype)


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
      self._step = None
    return obs, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class ActionRepeat:

  def __init__(self, env, amount):
    self._env = env
    self._amount = amount

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    done = False
    total_reward = 0
    current_step = 0
    while current_step < self._amount and not done:
      obs, reward, done, info = self._env.step(action)
      total_reward += reward
      current_step += 1
    return obs, total_reward, done, info


class NormalizeActions:

  def __init__(self, env):
    self._env = env
    self._mask = np.logical_and(
        np.isfinite(env.action_space.low),
        np.isfinite(env.action_space.high))
    self._low = np.where(self._mask, env.action_space.low, -1)
    self._high = np.where(self._mask, env.action_space.high, 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    return gym.spaces.Box(low, high, dtype=np.float32)

  def step(self, action):
    original = (action + 1) / 2 * (self._high - self._low) + self._low
    original = np.where(self._mask, original, action)
    return self._env.step(original)


class RewardObs:

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    assert 'reward' not in spaces
    spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
    return gym.spaces.Dict(spaces)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reward'] = reward
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reward'] = 0.0
    return obs

class Async:

  _ACCESS = 1
  _CALL = 2
  _RESULT = 3
  _EXCEPTION = 4
  _CLOSE = 5

  def __init__(self, ctor, strategy='process'):
    self._strategy = strategy
    if strategy == 'none':
      self._env = ctor()
    elif strategy == 'thread':
      import multiprocessing.dummy as mp
    elif strategy == 'process':
      import multiprocessing as mp
    else:
      raise NotImplementedError(strategy)
    if strategy != 'none':
      self._conn, conn = mp.Pipe()
      self._process = mp.Process(target=self._worker, args=(ctor, conn))
      atexit.register(self.close)
      self._process.start()
    self._obs_space = None
    self._action_space = None

  @property
  def observation_space(self):
    if not self._obs_space:
      self._obs_space = self.__getattr__('observation_space')
    return self._obs_space

  @property
  def action_space(self):
    if not self._action_space:
      self._action_space = self.__getattr__('action_space')
    return self._action_space

  def __getattr__(self, name):
    if self._strategy == 'none':
      return getattr(self._env, name)
    self._conn.send((self._ACCESS, name))
    return self._receive()

  def call(self, name, *args, **kwargs):
    blocking = kwargs.pop('blocking', True)
    if self._strategy == 'none':
      return functools.partial(getattr(self._env, name), *args, **kwargs)
    payload = name, args, kwargs
    self._conn.send((self._CALL, payload))
    promise = self._receive
    return promise() if blocking else promise

  def close(self):
    if self._strategy == 'none':
      try:
        self._env.close()
      except AttributeError:
        pass
      return
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      # The connection was already closed.
      pass
    self._process.join()

  def step(self, action, blocking=True):
    return self.call('step', action, blocking=blocking)

  def reset(self, blocking=True):
    return self.call('reset', blocking=blocking)

  def _receive(self):
    try:
      message, payload = self._conn.recv()
    except ConnectionResetError:
      raise RuntimeError('Environment worker crashed.')
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    raise KeyError(f'Received message of unexpected type {message}')

  def _worker(self, ctor, conn):
    try:
      env = ctor()
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          if not conn.poll(0.1):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACCESS:
          name = payload
          result = getattr(env, name)
          conn.send((self._RESULT, result))
          continue
        if message == self._CALL:
          name, args, kwargs = payload
          result = getattr(env, name)(*args, **kwargs)
          conn.send((self._RESULT, result))
          continue
        if message == self._CLOSE:
          assert payload is None
          break
        raise KeyError(f'Received message of unknown type {message}')
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      print(f'Error in environment process: {stacktrace}')
      conn.send((self._EXCEPTION, stacktrace))
    conn.close()

def parse(spec, options):
  """Parses a spec string given a list of options. This can use parametric options, e.g.
  parse(s, ['targetrew(\d*)x']). They can be accessed as e.g. targetrewNx.

  :param spec: a string separated by underscore, e.g. kitchen_microwave_fullstaterev_topviewcent.
  It will be parsed to determine whether any of the spec elements match an option.
  :param options: a list of regular expressions that will be matched against the spec elements. They will be matched  exactly.
  :return: a dictionary of truth values for each option, and a spec string with all parsed elements removed.
  """

  opt_names = [o.replace('(\d*)', 'N') for o in options]
  options = [re.compile('^' + o + '$') for o in options]

  # Parse
  spec_list = [s for s in spec.split('_') if any(o.search(s) for o in options)]

  def parse_option(o):
    match = any_obj(o.match(s) for s in spec_list)
    if not match:
      return False
    if len(match.groups()) == 0:
      return True
    return match.groups()[0]

  matches = [parse_option(o) for o in options]
  keys = dict((n, m) for n, m in zip(opt_names, matches))

  # Remove keys from name
  spec = '_'.join([s for s in spec.split('_') if s not in spec_list])
  return keys, spec


def any_obj(l):
  l = list(l)
  nonzero = np.nonzero(l)[0]
  if len(nonzero) > 0:
    return l[nonzero[0]]
  else:
    return False


def set_camera(cam, azimuth=None, elevation=None, distance=None, lookat=None):
    """ Sets camera parameters """
    if azimuth:
        cam.azimuth = azimuth
    if elevation:
        cam.elevation = elevation
    if distance:
        cam.distance = distance
    if lookat:
        for i in range(3):
            cam.lookat[i] = lookat[i]
