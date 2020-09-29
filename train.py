#!/usr/bin/env python3
import argparse
import random

import numpy as np
import torch
from rl.agents import TD3
from cassie.envs import cassie_standing, cassie_walking, cassie_jumping
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cassie Curriculum Learning')

    # Environment parameters
    parser.add_argument('--env', '-e', type=int, default=0, dest='env',
                        help='Cassie environment: [0] Standing, [1] Walking [2] Jumping(default: Standing)')
    parser.add_argument('--simrate', type=int, default=60,
                        help='Simulation rate in Hz (default: 60)')
    parser.add_argument('--no_clock', action='store_false', default=True, dest='clock',
                        help='Disables clock')
    parser.add_argument('--no_state_est', action='store_false', default=True, dest='state_est',
                        help='Disables state estimator')
    parser.add_argument('--rcut', '-r', nargs='+', type=float, default=[0.5], dest='rcut',
                        help='Ends an episode if a step reward falls below this threshold. '
                             'Enter two values [initial, final] cutoff to activate termination curriculum '
                             '(default: 0.5)')
    parser.add_argument('--tw', type=float, default=1.,
                        help='Weight multiplied to the action offset added to the policy action (default: 0.0)')
    parser.add_argument('--forces', '-f', nargs='+', type=float, default=(0., 0., 0.),
                        help='Forces applied to the pelvis i.e. [x, y, z] (default: (0, 0, 0) )')
    parser.add_argument('--force_fq', type=int, default=100,
                        help='Timestep frequency of forces applied to the pelvis (default: 100)')
    parser.add_argument('--fall_height', type=float, default=0.7,
                        help='Height in meters that the environment considers falling when it goes below this parameter'
                             ' (default: 0.7)')
    parser.add_argument('--speed', nargs='+', type=float, default=(0, 1),
                        help='Min and max speeds in m/s (default: [0, 1])')
    parser.add_argument('--power_threshold', type=int, default=150,
                        help='Power threshold to train on. Measured in Watts (default: 150)')
    parser.add_argument('--config', action='store', default="cassie/cassiemujoco/cassie.xml",
                        help='Path to the configuration file to load in the simulation (default: '
                             'cassie/cassiemujoco/cassie.xml )')
    parser.add_argument('--reduced_input', action='store_true', default=False,
                        help='Trains with inputs that are directly measured only (default: False)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Activates reward debug (default: False)')

    # Training parameters
    parser.add_argument('--training_steps', type=float, default=1e6,
                        help='Total timesteps to run the training (default: 1e6)')
    parser.add_argument('--eps_steps', type=int, default=30,
                        help='Number of timesteps in each episode, 1 cycle is about 24 timesteps (default: 30)')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='Evaluate policy every specified timestep intervals (default: 100)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Creates a seed to the specified value (default: None)')
    parser.add_argument('--expl_noise', type=float, default=0.1,
                        help='Upper bound on added noise added to the policy output for exploration (default=0.1)')
    parser.add_argument('--reset_ratio', type=float, default=0.7,
                        help='Ratio for phase and full reset. Value closer to one does more phase resets (default=0.7)')

    # File and Logging parameters
    parser.add_argument('--save', '-s', action='store_true', default=False, dest='save',
                        help='Saves the policy to the results folder (default: False)')
    parser.add_argument('--tag', action='store', default='',
                        help='Adds a tag at the end of agent_id (default: None)')
    parser.add_argument('--load', '-l', action='store', default=None, dest='load',
                        help='Provide path to existing model to load it (default=None)')
    parser.add_argument('--tensorboard', action='store_true', default=False, dest='tensorboard',
                        help='Log data using tensorboard. May max out memory (default=False)')

    # Algorithm Parameters
    parser.add_argument('--algo', action='store', default='TD3',
                        help='Name of algorithm to use [TD3, SAC] (default: TD3)')
    parser.add_argument('--hidden', type=float, nargs='+', default=(256, 256),
                        help='Size of the 2 hidden layers (default=[256, 256])')
    parser.add_argument('--alr', type=float, default=5e-5,
                        help='Actor learning rate(s) (default=5e-5)')
    parser.add_argument('--clr', type=float, default=8e-5,
                        help='Critic learning rate(s) (default=[8e-5])')
    parser.add_argument('--buffer', type=float, default=1e6,
                        help='Replay buffer size (default=1e6)')
    parser.add_argument('--batch', type=int, default=1024,
                        help='Batch size (default=1024)')
    parser.add_argument('--tau', '-t', type=float, default=1e-3,
                        help='Target network update rate (default=1e-3)')
    parser.add_argument('--discount', type=float, default=0.99,
                        help='Discount factor (default=0.99)')
    parser.add_argument('--start_steps', type=int, default=10000,
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--network_init', action='store_true', default=False,
                        help='Enables network initialization')

    # TD3 Specific Parameters
    parser.add_argument('--update_fq', type=int, default=2, dest='update_fq',
                        help='Policy update frequency (default=2)')
    parser.add_argument('--policy_noise', type=float, default=0.35,
                        help='Noise added to target networks during critic update (default=0.35)')

    # SAC Specific Parameters
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Temperature parameter α determines the relative importance of the entropy\
                                   term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True,
                        help='Automaically adjust α, if True alpha parameter is ignored (default: True)')
    parser.add_argument('--target_update_interval', type=int, default=1,
                        help='Value target update per no. of updates per step (default: 1)')

    args = parser.parse_args()

    if args.seed is not None:
        # initialize seeds
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # create envs list
    envs = (('Standing', cassie_standing.CassieEnv), ('Walking', cassie_walking.CassieEnv),
            ('Jumping', cassie_jumping.CassieEnv))

    # create agent id
    agent_id = '{}[RC{}TW{}]_{}[ALR{}CLR{}BATCH{}GAMMA{}]_Training[TS{}ES{}S{}RST{}SPD{}PWR{}FH{}CLK{}RI{}]{}'.format(
        envs[args.env][0],
        args.rcut,
        args.tw,
        args.algo.upper(),
        args.alr,
        args.clr,
        args.batch,
        args.discount,
        int(args.training_steps),
        args.eps_steps,
        args.seed,
        args.reset_ratio,
        args.speed,
        args.power_threshold,
        args.fall_height,
        args.clock,
        args.reduced_input,
        args.tag)

    # create SummaryWriter instance to log information
    writer = SummaryWriter('runs/{}. {}/Running/{}'.format(int(args.env + 1),
                                                           envs[args.env][0],
                                                           agent_id), flush_secs=60) if args.tensorboard else None

    # initialize environment
    env = envs[args.env][1](simrate=args.simrate,
                            clock_based=args.clock,
                            state_est=args.state_est,
                            reward_cutoff=args.rcut[0],
                            target_action_weight=args.tw,
                            fall_height=args.fall_height,
                            forces=args.forces,
                            force_fq=args.force_fq,
                            min_speed=args.speed[0],
                            max_speed=args.speed[1],
                            power_threshold=args.power_threshold,
                            reduced_input=args.reduced_input,
                            debug=args.debug,
                            config=args.config, )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # initialize agent
    agent = TD3.Agent(args.algo,
                      state_dim,
                      action_dim,
                      max_action,
                      hidden_dim=args.hidden,
                      actor_lr=args.alr,
                      critic_lr=args.clr,
                      discount=args.discount,
                      tau=args.tau,
                      policy_noise=args.policy_noise,
                      random_action_steps=args.start_steps,
                      capacity=args.buffer,
                      batch_size=args.batch,
                      policy_update_freq=args.update_fq,
                      chkpt_pth=args.load,
                      init_weights=args.network_init,
                      termination_curriculum=args.rcut if len(args.rcut) == 2 else None,
                      writer=writer)

    # run training
    agent.train(env,
                int(args.training_steps),
                args.eps_steps,
                args.eval_interval,
                expl_noise=args.expl_noise,
                filename='{}. {}/{}'.format(int(args.env + 1),
                                            envs[args.env][0],
                                            agent_id) if args.save else None,
                reset_ratio=args.reset_ratio, )

    if writer:
        # cleanup
        writer.close()
