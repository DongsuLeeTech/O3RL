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

from Algos.IQL import ImplicitQLearning
from Utils.utils import *
from tqdm import tqdm

import uuid
import wandb

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Parse argument used when running a Flow simulation.",
    epilog="python simulate.py EXP_CONFIG")
# required input parameters
parser.add_argument(
    'exp_config', type=str,
)  # Name of the experiment configuration file
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

# finetune RL parameter
parser.add_argument('--model-path', type=str)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--offline-data', action='store_true')
parser.add_argument('--num-evaluations', type=int, default=5)
parser.add_argument('--freezeQ', action='store_true')

# Actor critic parameter
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--batch', type=int, default=64)  # batch size to update
parser.add_argument('--discount', type=float, default=0.99,)  # discounted factor
parser.add_argument('--target-update-interval', type=int, default=2)
parser.add_argument('--l2_rate', type=float, default=1e-3,)
parser.add_argument('--eps', type=float, default=1e-08)
parser.add_argument('--actor_lr', type=float, default=1e-04,)
parser.add_argument('--q_lr', type=float, default=1e-04,)

# BCQ algorithm parameter
parser.add_argument("--alpha", default=0.7, type=float)
parser.add_argument("--v_lr", default=1e-04, type=float)
parser.add_argument("--beta", default=3.0, type= float)
parser.add_argument("--deterministic", action='store_true')

parser.add_argument('--project', default='ABS')
parser.add_argument('--group', default='ABS-IQL-fine-tuning')
parser.add_argument('--name', default='IQL-finetune')

args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args.device)
args.render = not args.no_render


def main(args, replay_buffer):
    # Import relevant information from the exp_config script.
    module = __import__(
        "exp_configs.rl.singleagent", fromlist=[args.exp_config])
    module_ma = __import__(
        "exp_configs.rl.multiagent", fromlist=[args.exp_config])

    # rl part
    if hasattr(module, args.exp_config):
        submodule = getattr(module, args.exp_config)
        multiagent = False
    elif hasattr(module_ma, args.exp_config):
        submodule = getattr(module_ma, args.exp_config)
        multiagent = True
    else:
        raise ValueError("Unable to find experiment config.")

    flow_params = submodule.flow_params

    import ray
    from ray.tune.registry import register_env
    try:
        from ray.rllib.agents.agent import get_agent_class
    except ImportError:
        from ray.rllib.agents.registry import get_agent_class

    alg_run = "PPO"
    agent_cls = get_agent_class(alg_run)
    config = deepcopy(agent_cls._default_config)

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json

    ray.init(num_cpus=16, object_store_memory=200 * 1024 * 1024)

    create_env, gym_name = make_create_env(params=flow_params, version=0)
    register_env(gym_name, create_env)
    agent = agent_cls(env=gym_name, config=config)

    # warmup correction
    if args.exp_config == 'MA_4BL':
        warmup_ts = 900
    elif args.exp_config == 'MA_5LC':
        warmup_ts = 125
    elif args.exp_config == 'UnifiedRing':
        warmup_ts = 90

    #  Load the Environment and Random Seed
    env = gym.make(gym_name)

    # Setup Random Seed
    env_set_seed(env, args.seed)

    num_inputs = 19
    num_actions = 2
    max_action = 1.0
    print(env.action_space)
    print('state size:', num_inputs)
    print('action size:', num_actions)

    # load Human Driving Data (NGSIM)
    buffer_name = f"{args.dataset}"
    setting = f"{args.dataset}_{args.seed}_{args.batch}"

    # offline buffer loading
    if args.offline_data:
        replay_buffer.load(f"./buffers/{args.road}/{buffer_name}")

    # Initialize and load policy and Q_net
    policy = ImplicitQLearning(args, num_inputs, num_actions, max_action, args.device)
    policy.load(args.model_path)

    # offline RL performance test
    print('----------------------------------------------------------------------------------------')
    print('-----------------Check Performance of Policy Pre-Trained by Offline RL------------------')

    timesteps = []
    evaluations = []
    for en in range(args.num_evaluations):
        env.seed(args.seed + en)
        tot_reward = 0.
        state, done = env.reset(), False
        horizon = 0
        while horizon <= args.horizon:
            action = policy.select_action(list(state.values()), deterministic=True)
            action = {list(state.keys())[0]: action}
            next_state, reward, done, _ = env.step(action)
            tot_reward += list(reward.values())[0]

            if done['__all__']:
                timesteps.append(env.unwrapped.k.vehicle.get_timestep(env.unwrapped.k.vehicle.get_ids()[1]) / 100)
                break
            else:
                pass

            state = next_state
            horizon += 1

        evaluations.append(tot_reward)

        eval_reward = np.mean(evaluations)
        eval_timestep = np.mean(timesteps) - warmup_ts
        eval_cor_reward = np.array(eval_timestep) + np.array(eval_reward)
        correction_reward = np.mean(eval_cor_reward)

    print('----------------------------------------------------------------------------------------')
    print('# avg.reward: {} # cor.reward: {}'.format(eval_reward, correction_reward))
    print('# average episode len: {}'.format(eval_timestep))
    print('----------------------------------------------------------------------------------------')

    wandb.log(
        {"vanilla_reward": eval_reward, "coorection_reward": correction_reward, "timesteps": eval_timestep},
        step=0)


    # Model Training
    print('----------------------------------------------------------------------------------------')
    print('---------------------------------Start Online Fine Tune----------------------------------')

    ts = 1.
    ep = 0.
    while ts <= args.epochs * args.horizon:
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

            if ts >= args.batch * 10:
                if args.freezeQ:
                    policy.freeze_value(replay_buffer)
                else:
                    policy.train(replay_buffer)

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

        ep += 1.

        print('----------------------------------------------------------------------------------------')
        print('# episode: {} # avg.reward: {} # cor.reward: {}'.format(ep, eval_reward, correction_reward))
        print('# velocity list: {} over {} evaluations'.format(velocity, args.num_evaluations))
        print('# average episode len: {}'.format(eval_timestep))
        print('----------------------------------------------------------------------------------------')

        wandb.log(
            {"vanilla_reward": eval_reward, "coorection_reward": correction_reward, "timesteps": eval_timestep},
            step=int(ts))


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
    seed_list = [5, 6, 7]
    env_list = ['final-medium']
    road_type = 'bottleneck'
    args.road = road_type
    args.dataset = env_list[0]
    print(f'--------------------Dataset: {args.dataset}--------------------')
    for j in seed_list:
        args.seed = j

        state_dim = 19
        action_dim = 2
        buffer_name = f"{args.dataset}"
        set_seed(args.seed)

        if args.freezeQ:
            args.name = f"{args.name}-freezeQ-OffSeed{args.model_path[-1]}-Seed{args.seed}-{args.road}-{args.dataset}-{str(uuid.uuid4())[:8]}"
        else:
            args.name = f"{args.name}-OffSeed{args.model_path[-1]}-Seed{args.seed}-{args.road}-{args.dataset}-{str(uuid.uuid4())[:8]}"
        config = vars(args)
        wandb_init(config)

        buffer_size = len(np.load(f"./buffers/{args.road}/{buffer_name}/reward.npy"))
        replay_buffer = ReplayBuffer(state_dim, action_dim, args.device, buffer_size)

        from datetime import datetime
        date = datetime.today().strftime("[%Y|%m|%d|%H:%M:%S]")
        args.log_dir = f"{date}-{args.road}-{buffer_name}-{args.seed}-finetune"
        os.mkdir(f'./results/{args.log_dir}')

        print('-----------------------------------------------------')
        main(args, replay_buffer)
        wandb.finish()
        args.name = 'IQL-finetune'
        print('-------------------DONE Fine-tune RL-------------------')

        import ray
        ray.shutdown()