import argparse
import numpy as np
import os
import pathlib
import pandas as pd
import sys

from glob import glob
from matplotlib import pyplot as plt


def eval_episodes(filenames):
    # Returns array of length <res>
    rew_list = []
    for filename in filenames:
        filename = pathlib.Path(filename).expanduser()
        with filename.open('rb') as f:
            episode = np.load(f)
            episode = {k: episode[k] for k in episode.keys()}
        if 'sparse_reward' in episode:
            rew_list.append(float(episode['sparse_reward'].sum() > 0))
        else:
            rew_list.append(episode['reward'].sum())

    rew = np.mean(rew_list)
    sem = np.std(rew_list) / np.sqrt(len(rew_list))
    return rew, sem

def main(config):
    indices = config.row_labels
    headers = config.col_labels
    df = pd.DataFrame(columns=headers)
    for j, task in enumerate(config.tasks):
        print(f'Evaluating {task}')
        for i, method in enumerate(config.methods):
            seeds = config.logdir.glob(f'{task}/{method}/*')
            rews = []
            for seed in seeds:
                if not seed.relative_to(config.logdir).parts[-1].isdigit():
                    continue
                filenames = sorted(glob(f'{seed}/{config.episodes_dir}/*.npz'))
                # Take the most recent episodes
                if config.num_episodes > len(filenames):
                    print(f"Total number of episodes ({len(filenames)}) less than expected ({config.num_episodes})")
                filenames = filenames[-config.num_episodes:]
                rew, sem = eval_episodes(filenames)
                rews.append(rew)
            mean = np.mean(rews)
            sem = np.std(rews) / np.sqrt(len(rews))
            if 'mw' in task:
                df.loc[indices[j], headers[i]] = f'{mean*100:.0f} $\\pm$ {sem*100:.0f}\\%'
                print(f'{config.methods[i]}:\t succ {mean:.2f}\tsem {sem:.2f}')
            else:
                df.loc[indices[j], headers[i]] = f'{mean:.0f} $\\pm$ {sem:.0f}'
                print(f'{config.methods[i]}:\t rew {mean:.2f}\tsem {sem:.2f}')
    print(df.transpose().to_latex(escape=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    dir = lambda x: pathlib.Path(x).expanduser()
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--logdir', type=dir, default='')
    parser.add_argument('--tasks', type=str, default='', nargs='+')
    parser.add_argument('--methods', type=str, default='', nargs='+')
    parser.add_argument('--row_labels', type=str, default='', nargs='+')
    parser.add_argument('--col_labels', type=str, default='', nargs='+')
    parser.add_argument('--episodes_dir', type=str, default='episodes')
    config = parser.parse_args()
    main(config)
