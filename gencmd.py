import argparse
import yaml

from pathlib import Path

def args2list(arg_dict):
    arg_list = []
    for name, val in arg_dict.items():
        arg_list.extend(['--' + name, str(val)])
    return arg_list

def main(args):
    # Load configuration file
    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exception:
            print(exception)

    # Gather arguments
    suite, task = args.task.split('_')
    cfg = config[suite]
    train_args = cfg['train']
    task_args = cfg['tasks'][task]
    method_args = cfg['planners'][args.method]

    run_args = {}
    run_args.update(train_args)
    run_args.update(task_args)
    run_args.update(method_args)
    if args.dense_mw:
        run_args.update(cfg['dense'])
    run_args['logdir'] = '/'.join([args.logdir, args.task, args.method, str(args.seed)])
    if args.eval:
        run_args['logdir_eval'] = '/'.join([args.logdir_eval, args.task, args.method, str(args.seed)])

    # Print command
    cmd = ' '.join(['python', 'train.py' if not args.eval else 'eval.py'] + args2list(run_args))
    print(cmd)

if __name__ == '__main__':
    boolean = lambda x: bool(['False', 'True'].index(x))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--method', type=str, default='latco')
    parser.add_argument('--task', type=str, default='mw_reach')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='logdir')
    parser.add_argument('--eval', type=boolean, default=False)
    parser.add_argument('--dense_mw', type=boolean, default=False)
    parser.add_argument('--logdir_eval', type=str, default='logdir_eval')
    args = parser.parse_args()
    main(args)
