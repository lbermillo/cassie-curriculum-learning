#!/usr/bin/env python3

import argparse

import torch
from cassie.envs import cassie_standing, cassie_walking
from rl.agents import TD3

# use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cassie Policy Evaluation')

    # Environment parameters
    parser.add_argument('--env', '-e', type=int, default=0, dest='env',
                        help='Cassie environment: [0] Standing, [1] Walking (default: Standing)')
    parser.add_argument('--simrate', type=int, default=60,
                        help='Simulation rate in Hz (default: 60)')
    parser.add_argument('--no_clock', action='store_false', default=True, dest='clock',
                        help='Disables clock and uses reference trajectories')
    parser.add_argument('--rcut', '-r', type=float, default=0.3, dest='rcut',
                        help='Ends an episode if a step reward falls below this threshold (default: 0.3)')
    parser.add_argument('--tw', type=float, default=1.,
                        help='Weight multiplied to the action offset added to the policy action (default: 1.0)')
    parser.add_argument('--forces', '-f', nargs='+', type=float, default=(0., 0., 0.),
                        help='Forces applied to the pelvis i.e. [x, y, z] (default: (0, 0, 0) )')
    parser.add_argument('--force_fq', type=int, default=10,
                        help='Frequency forces applied to the pelvis (default: 10 timesteps)')
    parser.add_argument('--min_speed', nargs='+', type=float, default=(0, 0, 0),
                        help='min speeds in m/s (default: [0, 0, 0])')
    parser.add_argument('--max_speed', nargs='+', type=float, default=(0, 0, 0),
                        help='max speeds in m/s (default: [0, 0, 0])')
    parser.add_argument('--power_threshold', type=int, default=150,
                        help='Power threshold to train on. Measured in Watts (default: 150)')
    parser.add_argument('--config', action='store', default="cassie/cassiemujoco/cassie.xml",
                        help='Path to the configuration file to load in the simulation (default: '
                             'cassie/cassiemujoco/cassie.xml )')
    parser.add_argument('--reduced_input', action='store_true', default=False,
                        help='Trains with inputs that are directly measured only (default: False)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Activates reward debug (default: False)')

    # Evaluation parameters
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Number of times to run the evaluation (default: 10)')
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='Number of timesteps in each episode, 1 cycle is about 24 timesteps (default: 30)')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Renders Cassie simulation (default: False)')
    parser.add_argument('--print_stats', action='store_false', default=True,
                        help='Prints episode rewards (default: True)')
    parser.add_argument('--reset_ratio', type=float, default=0,
                        help='Ratio for frequency of applying perturbations (default=0)')
    parser.add_argument('--use_phase', action='store_true', default=False,
                        help='Enables phase resets')

    # Algorithm Parameters
    parser.add_argument('--algo', action='store', default='TD3',
                        help='Name of algorithm to use [TD3, SAC] (default: TD3)')

    # File parameters
    parser.add_argument('--load', '-l', action='store', default=None, dest='load', required=True,
                        help='Provide path to existing model to load it (default=None)')

    args = parser.parse_args()

    # create envs list
    envs = (('Standing', cassie_standing.CassieEnv), ('Walking', cassie_walking.CassieEnv))

    env = envs[args.env][1](simrate=args.simrate,
                            clock_based=args.clock,
                            reward_cutoff=args.rcut,
                            target_action_weight=args.tw,
                            forces=args.forces,
                            force_fq=args.force_fq,
                            min_speed=args.min_speed,
                            max_speed=args.max_speed,
                            power_threshold=args.power_threshold,
                            reduced_input=args.reduced_input,
                            config=args.config,
                            debug=args.debug)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # initialize agent
    agent = TD3.Agent(args.algo,
                      state_dim,
                      action_dim,
                      max_action,
                      chkpt_pth=args.load, )

    agent.evaluate(env,
                   eval_eps=args.eval_episodes,
                   max_steps=args.eval_steps,
                   render=args.render,
                   print_stats=args.print_stats,
                   reset_ratio=args.reset_ratio,
                   use_phase=args.use_phase, )
