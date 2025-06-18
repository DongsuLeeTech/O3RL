import os
import sys
import gym
import json
import pickle
import random
import argparse
import numpy as np
from copy import deepcopy

import torch

from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.utils.registry import make_create_env

from Algos.AWAC import AWAC
from Utils.utils import *
from tqdm import tqdm

import uuid
import wandb

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Parse argument used when running a Flow simulation.",
    epilog="python simulate.py EXP_CONFIG")
# required input parameters
parser.add_argument(  # for rllib
    '--algorithm', type=str, default="PPO",
)  # choose algorithm in order to use
parser.add_argument(
    '--num_cpus', type=int, default=1,
)  # How many CPUs to use
parser.add_argument(  # batch size
    '--rollout_size', type=int, default=100,
)  # How many steps are in a training batch.
parser.add_argument(
    '--checkpoint_path', type=str, default=None,
)  # Directory with checkpoint to restore training from.
parser.add_argument(
    '--no_render',
    action='store_true',
)  # Specifies whether to run the simulation during runtime.

# network and dataset setting
parser.add_argument('--seed', type=int, default=0,)  # random seed
parser.add_argument('--dataset', type=str, default=None)  # path to datset
parser.add_argument('--load_model', type=str, default=None,)  # path to load the saved model
parser.add_argument('--logdir', type=str, default='./results/',)  # tensorboardx logs directory

# Fine tune parameter
parser.add_argument('--fine-tune', action='store_true')
parser.add_argument('--num', type=int)
parser.add_argument('--buffers', type=int, default=1e6)
parser.add_argument('--horizon', type=int, default=3000)
parser.add_argument('--max-ts', type=int, default=int(1e6))
parser.add_argument('--off_data', default=False, action='store_true')
parser.add_argument('--warmup_sample', default=False, action='store_true')
parser.add_argument('--model-path', type=str)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--num-evaluations', type=int, default=5)
parser.add_argument('--fl_init', default=False, action='store_true')
parser.add_argument('--logstd_init', default=False, action='store_true')
parser.add_argument('--Q_init', default=False, action='store_true')

# OfflineRL parameter
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--discount', type=float, default=0.99,)
parser.add_argument('--targ-update-freq', type=float, default=2)

# AWAC algorithm parameter
parser.add_argument('--actor-lr', type=float, default=3e-04,)
parser.add_argument('--q-lr', type=float, default=3e-04,)
parser.add_argument('--awac-lambda', type=float, default=0.3333)
parser.add_argument('--exp-adv-max', type=float, default=100.0)

# ABC algorithm parameter
parser.add_argument('--q_epochs', type=int, default=2000)
parser.add_argument('--q_train', action='store_true')
parser.add_argument('--warm_start', action='store_true')
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--alpha_lr', type=float, default=1e-04)

parser.add_argument('--project', default='ABS')
parser.add_argument('--group', default='ABS-MT-AWAC-fine-tuning')
parser.add_argument('--name', default='MT-AWAC-finetune')

args = parser.parse_args()
args.device = torch.device("cpu")
print(args.device)
args.render = not args.no_render

fullname = args.name
group = args.group

if args.fl_init:
    if args.logstd_init:
        fullname = fullname + '-only_std_init'
        group = group + '-only_std_init'
    else:
        fullname = fullname + '-final_layer_init'
        group = group + '-final_layer_init'

if args.q_train:
    fullname = fullname + '-q_train'
    group = group + '-q_train'

if args.warm_start:
    fullname = fullname + '-warm_start'
    group = group + '-warm_start'

if args.Q_init:
    fullname = fullname + '-Q_init'
    group = group + '-Q_init'

if not args.off_data:
    fullname = fullname + '-wo_offD'
    group = group + '-wo_offD'

args.name = fullname
args.group = group

def main(args, replay_buffer):
    # Import relevant information from the exp_config script.
    module_list = []
    for i in ['MA_4BL', 'MA_5LC', 'UnifiedRing']:
        module_ma = __import__(
            "exp_configs.rl.multiagent", fromlist=[i]
        )
        module_list.append(module_ma)

    # rl part
    submodule_list = []
    for i, j in zip(module_list, ['MA_4BL', 'MA_5LC', 'UnifiedRing']):
        hasattr(i, j)
        submodule = getattr(i, j)
        multiagent = True
        submodule_list.append(submodule)

    flow_param_list = []
    for i in submodule_list:
        flow_param_list.append(i.flow_params)
    # flow_params = submodule.flow_params

    import ray
    from ray.tune.registry import register_env
    try:
        from ray.rllib.agents.agent import get_agent_class
    except ImportError:
        from ray.rllib.agents.registry import get_agent_class

    alg_run = "PPO"
    agent_cls = get_agent_class(alg_run)
    config = deepcopy(agent_cls._default_config)

    ray.init(num_cpus=16, object_store_memory=200 * 1024 * 1024)
    create_env_list = []
    gym_name_list = []
    for flow_params in flow_param_list:
        create_env, gym_name = make_create_env(params=flow_params, version=0)
        create_env_list.append(create_env)
        gym_name_list.append(gym_name)
        register_env(gym_name, create_env)
        agent = agent_cls(env=gym_name, config=config)

    #  Load the Environment and Random Seed
    env_list = []
    for i in gym_name_list:
        env = gym.make(i)
        env_list.append(env)
        # Setup Random Seed
        env_set_seed(env, args.seed)

        num_inputs = 19
        num_actions = 2
        max_action = 1.0
        print(env.action_space)
        print('state size:', num_inputs)
        print('action size:', num_actions)

    # Initialize and load policy and Q_net
    policy = AWAC(args, num_inputs, num_actions, max_action)
    policy.load(args.model_path, args)

    # final layer initialization
    if args.fl_init:
        policy.actor.init_layer(num_actions, args)

    # Replay buffer
    buffer_name = f"{args.dataset}"
    if args.off_data:
        replay_buffer.off_data_load(f"./buffers/{args.road}/{buffer_name}")

    # offline RL performance test
    print('----------------------------------------------------------------------------------------')
    print('-----------------Check Performance of Policy Pre-Trained by Offline RL------------------')

    for env, road in zip(env_list, ['MA_4BL', 'MA_5LC', 'UnifiedRing']):
        print(env)
        if road == 'MA_4BL':
            warmup_ts = 900
        elif road == 'MA_5LC':
            warmup_ts = 125
        elif road == 'UnifiedRing':
            warmup_ts = 90

        evaluations = []
        velocity = []
        timesteps = []

        for _ in range(args.num_evaluations):
            env.seed(args.seed + 100)
            tot_reward = 0.
            state, done = env.reset(), False

            episode_vel = []
            ts = 0
            while ts <= args.max_ts:
                if args.render:
                    env.render()
                action = policy.select_action(list(state.values()))
                episode_vel.append(list(state.values())[0])
                next_state, reward, done, _ = env.step({list(state.keys())[0]: action})
                tot_reward += list(reward.values())[0]

                if args.warmup_sample:
                    replay_buffer.add(list(state.values())[0], action,
                                      list(next_state.values())[0], list(reward.values())[0], done['__all__'] * 1)

                if done['__all__']:
                    timesteps.append(
                        env.unwrapped.k.vehicle.get_timestep(env.unwrapped.k.vehicle.get_ids()[1]) / 100)
                    break
                else:
                    pass

                state = next_state
            velocity.append(np.mean(episode_vel))
            evaluations.append(tot_reward)

        eval_reward = np.mean(evaluations)
        eval_timestep = np.mean(timesteps) - warmup_ts
        correction_reward = np.mean(np.array(evaluations) + np.array(timesteps) - warmup_ts)

        print('----------------------------------------------------------------------------------------')
        print('# itr: {} # avg.reward: {} # cor.reward: {}'.format(road, eval_reward, correction_reward))
        print('# velocity list: {} over {} evaluations'.format(velocity, args.num_evaluations))
        print('# average episode len: {}'.format(eval_timestep))
        print('----------------------------------------------------------------------------------------')

        wandb.log(
            {f"{road}-vanilla_reward": eval_reward, f"{road}-correction_reward": correction_reward, f"{road}-timesteps": eval_timestep},
            step=0)

    ## Offline Phase 2
    if args.q_train:
        print('')
        print('----------------------------------------------------------------------------------------')
        print('---------------------------------Start Offline Phase 2----------------------------------')
        for ep in range(1, args.q_epochs + 1):
            policy.q_train(replay_buffer)
        print('----------------------------------------------------------------------------------------')
        print('----------------------------------End Offline Phase 2-----------------------------------')
        print('')

    # Model Training
    print('----------------------------------------------------------------------------------------')
    print('---------------------------------Start Online Fine Tune----------------------------------')

    ts = 1.
    for ep in range(1, args.epochs+1):
        for env, road in zip(env_list, ['MA_4BL', 'MA_5LC', 'UnifiedRing']):
            if road == 'MA_4BL':
                warmup_ts = 900
            elif road == 'MA_5LC':
                warmup_ts = 125
            elif road == 'UnifiedRing':
                warmup_ts = 90

            env.seed(args.seed)
            tot_reward = 0.
            state, done = env.reset(), False

            velocity = []
            timesteps = []
            evaluations = []
            episode_vel = []

            horizon = 0.
            while horizon < args.horizon:
                if args.render:
                    env.render()
                action = policy.select_action(list(state.values()), deterministic=False)
                episode_vel.append(list(state.values())[0])
                next_state, reward, done, _ = env.step({list(state.keys())[0]: action})
                tot_reward += list(reward.values())[0]
                replay_buffer.add(list(state.values())[0], action,
                                  list(next_state.values())[0], list(reward.values())[0], done['__all__'] * 1)

                if not args.warm_start:
                    policy.train(replay_buffer)
                else:
                    policy.online_train(replay_buffer)

                if done['__all__']:
                    timesteps.append(
                        env.unwrapped.k.vehicle.get_timestep(env.unwrapped.k.vehicle.get_ids()[1]) / 100)
                    break
                else:
                    pass

                ts += 1.
                horizon += 1.
                state = next_state

            velocity.append(np.mean(episode_vel))
            evaluations.append(tot_reward)

            eval_reward = np.mean(evaluations)
            eval_timestep = np.mean(timesteps) - warmup_ts
            correction_reward = np.mean(np.array(evaluations) + np.array(timesteps) - warmup_ts)

            print('----------------------------------------------------------------------------------------')
            print('# episode: {} # avg.reward: {} # cor.reward: {}'.format(ep, eval_reward, correction_reward))
            print('# velocity list: {} over {} evaluations'.format(velocity, args.num_evaluations))
            print('# average episode len: {}'.format(eval_timestep))
            print('----------------------------------------------------------------------------------------')

            wandb.log(
                {f"{road}-vanilla_reward": eval_reward, f"{road}-correction_reward": correction_reward, f"{road}-timesteps": eval_timestep},
                step=int(ep))

    policy.save(f'./results/{args.log_dir}/{args.log_dir}')

def save_checkpoint(state, filename):
    torch.save(state, filename)

def wandb_init(config:dict) -> None:
    wandb.init(
        config=config,
        project=config['project'],
        group=config['group'],
        name=config['name'],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()

if __name__=="__main__":
    seed_list = [4, 5, 6]
    env_list = ['final-medium', 'medium', 'final-random', 'medium-random']
    args.road = 'mixed_env'
    for i in env_list:
        args.dataset = i
        for j in seed_list:
            args.seed = j
            args.model_path = './model/MT-AWAC/' + i + f'/Seed{j - 3}/'

            state_dim = 19
            action_dim = 2
            buffer_name = f"{args.dataset}"
            set_seed(args.seed)

            # args.name = f"{args.name}-OffSeed{args.model_path[-1]}-Seed{args.seed}-{args.road}-{args.dataset}-{str(uuid.uuid4())[:8]}"
            args.name = f"{args.name}-Seed{args.seed}-{args.road}-{args.dataset}-{str(uuid.uuid4())[:8]}"
            config = vars(args)
            wandb_init(config)

            buffer_size = len(np.load(f"./buffers/{args.road}/{buffer_name}/reward.npy"))
            replay_buffer = ReplayBuffer(state_dim, action_dim, args.device, buffer_size, args.off_data)

            from datetime import datetime
            date = datetime.today().strftime("[%Y|%m|%d|%H:%M:%S]")
            args.log_dir = f"{date}-{args.road}-{buffer_name}-{args.seed}-finetune"
            os.mkdir(f'./results/{args.log_dir}')

            print('-----------------------------------------------------')
            main(args, replay_buffer)
            wandb.finish()
            args.name = fullname
            print('-------------------DONE Fine-tune RL-------------------')

            import ray
            ray.shutdown()