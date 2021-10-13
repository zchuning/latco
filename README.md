## Model-Based Reinforcement Learning via Latent-Space Collocation

####  [[Project Website]](https://orybkin.github.io/latco/) [[Sparse MetaWorld Code]](https://github.com/zchuning/metaworld) [[Talk (5min)]](https://www.youtube.com/watch?v=skc0e4KYNcw) [[Paper]](https://arxiv.org/pdf/2106.13229.pdf)

[Oleh Rybkin*<sup>1</sup>](https://www.seas.upenn.edu/~oleh/), [Chuning Zhu*<sup>1</sup>](https://www.seas.upenn.edu/~zchuning/), [Anusha Nagabandi<sup>2</sup>](https://www.linkedin.com/in/anusha-nagabandi-a4923bba), [Kostas Daniilidis<sup>1</sup>](http://www.cis.upenn.edu/~kostas/), [Igor Mordatch<sup>3</sup>](https://twitter.com/imordatch), [Sergey Levine<sup>4</sup>](https://people.eecs.berkeley.edu/~svlevine/)<br/>
(&#42; equal contribution)

<sup>1</sup>University of Pennsylvania </br> <sup>2</sup>Covariant </br> <sup>3</sup>Google Brain</br> <sup>4</sup>UC Berkeley </br>

<a href="https://orybkin.github.io/latco/">
<p align="center">
<img src="https://github.com/orybkin/latco/blob/main/resources/teaser.gif" width="800">
</p>
</img></a>

This is a TF2 implementation for our latent-space collocation (LatCo) agent for model-based reinforcement learning. LatCo is a visual model-based reinforcement learning method that can solve long-horizon tasks by optimizing sequences of latent states, instead of optimizing actions directly as done in shooting methods such as visual foresight and planet. Optimizing latent states allows LatCo to quickly discover the high-reward region and construct effective plans even for complex multi-stage tasks that shooting methods do not solve.


## Instructions

#### Setting up repo
```
git clone https://github.com/zchuning/latco.git
cd latco
git submodule update --init --recursive
```


#### Dependencies

- Install [Mujoco](https://www.roboti.us/index.html)
- Install dependencies

```
pip install --user numpy==1.19.2 cloudpickle==1.2.2 tensorflow-gpu==2.2.0 tensorflow_probability==0.10.0
pip install --user gym imageio pandas pyyaml matplotlib
pip install --user mujoco-py (optional - to run Sparse MetaWorld or Pointmass)
pip install --user dm-control (optional - to run DM control)
pip install --user -e metaworldv2
```

The code is tested on Ubuntu 20.04, with CUDNN 7.6.5, CUDA 10.1, and Python 3.8


#### Commands

Train LatCo agent on the Reaching task:

```
python train.py --collect_sparse_reward True --use_sparse_reward True --task mw_SawyerReachEnvV2 --planning_horizon 30 --mpc_steps 30 --agent latco --logdir logdir/mw_reach/latco/0
```

Evaluate LatCo agent:

```
python eval.py --collect_sparse_reward True --use_sparse_reward True --task mw_SawyerReachEnvV2 --planning_horizon 30 --mpc_steps 30 --agent latco --logdir logdir/mw_reach/latco/0 --logdir_eval logdir_eval/mw_reach/latco/0 --n_eval_episodes 10
```

To run the offline+fine-tune experiments, download the released offline data for the [Hammer](https://drive.google.com/file/d/1nrswj4p3ZdHjdB6iM5LpHjkZrs2OXNco/view?usp=sharing) and the [Thermos](https://drive.google.com/file/d/1lRSpwrqjQe-KveYZ6XV1OC1V8HIYYJKD/view?usp=sharing) tasks. Train LatCo agent on the Thermos task with offline data (data path specified by `--offline_dir`):

```
python train.py --prefill 0 --action_repeat 2 --collect_sparse_reward True --use_sparse_reward True --task mw_SawyerStickPushEnvV2 --planning_horizon 50 --mpc_steps 25 --agent latco --logdir logdir/mw_thermos/latco/0 --offline_dir logdir/mw_thermos/offline/episodes
```

For convenience, we include a script for automatically generating training and evaluation commands with hyperparameters from the paper. To generate training command, run:

```
python gencmd.py --task mw_reach --method latco
```

To generate evaluation command, run:

```
python gencmd.py --task mw_reach --method latco --eval True
```

`--task` can be one of `mw_reach, mw_button, mw_window, mw_drawer, mw_push, mw_hammer, mw_thermos, dmc_reacher, dmc_cheetah, dmc_quadruped`. `--method` can be one of `latco, planet, mppi, shooting_gd, shooting_gn, platco`. To replicate ablation experiments, set `--method` to be one of `latco_no_constraint, latco_no_relax, latco_first_order, image_colloc`. To run dense reward metaworld tasks, add `--dense_mw True`.

Generate plots:

```
python plot.py --indir ./logdir --outdir ./plots --xaxis step --yaxis train/success --bins 3e4
```

Tensorboard:

```
tensorboard --logdir ./logdir
```


#### Troubleshooting


By default, the mujoco rendering for Sparse MetaWorld will use glfw. With egl rendering, gpu 0 will be used by default. You can change these with the following environment variables
```
export MUJOCO_RENDERER='egl'
export GL_DEVICE_ID=1
```

If you get `ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject`, chances are, your mujoco-py was not installed properly. You can fix it with the following

```
pip uninstall mujoco-py
pip install mujoco-py --no-cache-dir --no-binary :all: --no-build-isolation
```


## Code structure
- `latco.py` contains the LatCo agent. It inherits `planning_agent`.
- `planners/probabilistic_latco.py` contains the Gaussian LatCo agent. It inherits `planning_agent`.
- `planners/gn_solver.py` is a Gauss-Newton optimizer which leverages the block-tridiagonal structure of the Jacobian to speed up computation. This file is currently unavailable as it is going through the open-sourcing process. It will be available shortly.
- `base_agent.py` is a barebone agent with an RSSM model, modified from the Dreamer agent. `planning_agent.py` inherits `base_agent` and is inherited by all methods in `planners`.
- `planners` contains planning agents such as shooting cem, shooting gd, and a few other variants.
- `envs/sparse_metaworld.py` is the wrapper for the Sparse MetaWorld benchmark. The benchmark itself is a submodule `metaworldv2`.
- `envs/pointmass/pointmass_prob_env.py` is the pointmass lottery task.

#### Using Sparse MetaWorld environments

**WARNING!** The Sparse MetaWorld environments by default output dense reward. The sparse reward is in `info['success']`. A simple-to-use standalone environment code that outputs sparse reward is [here](https://github.com/zchuning/metaworld), please see usage instructions by that link.

#### Adding new environments

See `wrappers.py` as well as `envs/sparse_metaworld.py`, `envs/pointmass/pointmass_prob_env.py` for examples on how to add new environments. We follow a simple gym-like interface from the Dreamer repo.

Note you can also train our agent entirely offline on an existing dataset of episodes. To do this, use the `--offline_dir` argument to point to the dataset and set `--pretrain` to the desired number of training steps.

## Bibtex
If you find this code useful, please cite:

```
@inproceedings{rybkin2021latco,
  title={Model-Based Reinforcement Learning via Latent-Space Collocation},
  author={Rybkin, Oleh and Zhu, Chuning and Nagabandi, Anusha and Daniilidis, Kostas and Mordatch, Igor and Levine, Sergey},
  journal={Proceedings of the 38th International Conference on Machine Learning},
  year={2021}
}
```

## Acknowledgements

This codebase was built on top of [Dreamer](https://github.com/danijar/dreamer).
