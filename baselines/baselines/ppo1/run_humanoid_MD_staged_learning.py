#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import joblib
import tensorflow as tf
import numpy as np
from mpi4py import MPI
import os, errno

def callback(localv, globalv):
    if localv['iters_so_far'] % 10 != 0:
        return
    save_dict = {}
    variables = localv['pi'].get_variables()
    for i in range(len(variables)):
        cur_val = variables[i].eval()
        save_dict[variables[i].name] = cur_val

    save_dir = logger.get_dir() + '/' + (str(localv['env'].env.env.assist_schedule).replace(' ', ''))
    try:
        os.makedirs(save_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    joblib.dump(save_dict, save_dir+'/policy_params_'+ str(localv['iters_so_far'])+'.pkl', compress=True)
    joblib.dump(save_dict, logger.get_dir() + '/policy_params' + '.pkl', compress=True)

def train_mirror(args, num_timesteps):
    from baselines.ppo1 import mlp_mirror_policy, pposgd_mirror
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env)

    env.env._seed(args.seed+MPI.COMM_WORLD.Get_rank())
    env.env.assist_timeout = 100.0
    env.env.init_params(args)

    U.ALREADY_INITIALIZED = set()
    U.ALREADY_INITIALIZED.update(set(tf.global_variables()))

    def policy_fn(name, ob_space, ac_space):
        old_act_permute = [-86, 87, -88, 93, 94, 95, 96, 89, 90, 91, 92, 98, 97]
        mus_act_l = np.arange(43)
        mus_act_r = mus_act_l + 43
        mus_act_l[0] = 0.001
        act_permute = np.concatenate([mus_act_r, mus_act_l, old_act_permute])
        return mlp_mirror_policy.MlpMirrorPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                 hid_size=128, num_hid_layers=3, gmm_comp=1,
                                                 mirror_loss=True,
                                                 observation_permutation=np.array(
                                                     [0.0001, -1, 2, -3, -4,
                                                      11, 12, 13, 14, 15, 16, 5, 6, 7, 8, 9, 10, -17, 18, -19,
                                                      24, 25, 26, 27, 20, 21, 22, 23, \
                                                      28, 29, -30, 31, -32, -33,
                                                      40, 41, 42, 43, 44, 45, 34, 35, 36, 37, 38, 39, -46, 47, -48,
                                                      53, 54, 55, 56, 49, 50, 51, 52, 58, 57, 59]),
                                                 action_permutation=act_permute)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)
    env.seed(args.seed+MPI.COMM_WORLD.Get_rank())
    gym.logger.setLevel(logging.WARN)

    previous_params = None
    iter_num = 0
    last_iter = False

    # if initialize from previous runs
    #previous_params = joblib.load('')
    #env.env.env.assist_schedule = []

    joblib.dump(str(env.env.env.__dict__), logger.get_dir() + '/env_specs.pkl', compress=True)

    reward_threshold = None
    while True:
        if not last_iter:
            rollout_length_thershold = env.env.env.assist_schedule[2][0] / env.env.env.dt
        else:
            rollout_length_thershold = None
        opt_pi, rew = pposgd_mirror.learn(env, policy_fn,
                max_timesteps=num_timesteps,
                timesteps_per_batch=int(2500),
                clip_param=0.2, entcoeff=0.0,
                optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                gamma=0.99, lam=0.95, schedule='linear',
                callback=callback,
                sym_loss_weight=4.0,
                positive_rew_enforce=False,
                init_policy_params = previous_params,
                reward_drop_bound=500,
                rollout_length_thershold = rollout_length_thershold,
                policy_scope='pi' + str(iter_num),
                return_threshold = reward_threshold,
            )
        if iter_num == 0:
            reward_threshold = 0.7 * rew
        if last_iter:
            reward_threshold = None
        iter_num += 1

        opt_variable = opt_pi.get_variables()
        previous_params = {}
        for i in range(len(opt_variable)):
            cur_val = opt_variable[i].eval()
            previous_params[opt_variable[i].name] = cur_val
        # update the assist schedule
        for s in range(len(env.env.env.assist_schedule)-1):
            env.env.env.assist_schedule[s][1] = np.copy(env.env.env.assist_schedule[s+1][1])
        env.env.env.assist_schedule[-1][1][0] *= 0.75
        env.env.env.assist_schedule[-1][1][1] *= 0.75
        if env.env.env.assist_schedule[-1][1][0] < 5.0:
            env.env.env.assist_schedule[-1][1][0] = 0.0
        if env.env.env.assist_schedule[-1][1][1] < 5.0:
            env.env.env.assist_schedule[-1][1][1] = 0.0
        zero_assist = True
        for s in range(len(env.env.env.assist_schedule)-1):
            for v in env.env.env.assist_schedule[s][1]:
                if v != 0.0:
                    zero_assist = False
        print('Current Schedule: ', env.env.env.assist_schedule)
        if zero_assist:
            last_iter = True
            print('Entering Last Iteration!')
            env.env.add_flex_q_noise = False  # TODO

    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartHumanWalkerMD-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--HW_add_flex_noise_q', help='add noise to hip flex q?', type=bool, default=True)
    parser.add_argument('--HW_final_tv', help='final target velocity', type=float, default=1.5)
    parser.add_argument('--HW_tv_endtime', help='time to acc to final target velocity', type=float, default=1.2)
    parser.add_argument('--HW_energy_weight', help='energy pen weight', type=float, default=0.1)
    parser.add_argument('--HW_alive_bonus_rew', help='alive bonus weight', type=float, default=9.0)
    parser.add_argument('--HW_vel_reward_weight', help='velocity pen weight', type=float, default=4.5)
    parser.add_argument('--HW_side_devia_weight', help='side deviation pen weight', type=float, default=3.0)
    parser.add_argument('--HW_rot_pen_weight', help='rotation pen weight', type=float, default=0.0)
    parser.add_argument('--HW_abd_pen_weight', help='abdomen dof pen weight', type=float, default=1.0)
    parser.add_argument('--HW_spl_pen_weight', help='spine dof pen weight', type=float, default=0.5)
    parser.add_argument('--HW_angle_pen_weight', help='tilt angle pen weight', type=float, default=0.8)

    args = parser.parse_args()
    logger.reset()

    import datetime
    now = datetime.datetime.now()
    stampstring = now.isoformat()

    logdir = 'data/ppo_humanMD_'+stampstring[:16]+args.env+'_'+str(args.seed)
    for arg in vars(args):
        if arg[:3] == 'HW_':
            logdir += arg[2:6]
            logdir += '_'
            logdir += arg[-3:]
            logdir += str(getattr(args, arg))
    logger.configure(logdir)
    train_mirror(args, num_timesteps=int(5000*4*800))


if __name__ == '__main__':
    main()
