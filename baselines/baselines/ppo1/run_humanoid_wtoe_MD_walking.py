#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import joblib
import tensorflow as tf
import numpy as np
from mpi4py import MPI
import os, errno
import pprint
import shutil


def callback(localv, globalv):
    if localv['iters_so_far'] % 10 != 0:
        return
    save_dict = {}
    variables = localv['pi'].get_variables()
    for i in range(len(variables)):
        cur_val = variables[i].eval()
        save_dict[variables[i].name] = cur_val

    save_dir = logger.get_dir()
    try:
        os.makedirs(save_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    joblib.dump(save_dict, save_dir + '/policy_params_' + str(localv['iters_so_far']) + '.pkl', compress=True)
    joblib.dump(save_dict, logger.get_dir() + '/policy_params' + '.pkl', compress=True)


def train_mirror(args, num_timesteps):
    from baselines.ppo1 import mlp_mirror_policy, mlp_mirror_norms_policy, pposgd_mirror
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = gym.make(args.env)

    env.env._seed(args.seed + MPI.COMM_WORLD.Get_rank())
    env.env.init_params(args)

    U.ALREADY_INITIALIZED = set()
    U.ALREADY_INITIALIZED.update(set(tf.global_variables()))

    obs_per = np.array([0.0001, -1, 2, -3, -4,
                        11, 12, 13, 14, 15, 16, 5, 6, 7, 8, 9, 10, -17, 18, -19,
                        24, 25, 26, 27, 20, 21, 22, 23,
                        28, 29, -30, 31, -32, -33,
                        40, 41, 42, 43, 44, 45, 34, 35, 36, 37, 38, 39, -46, 47, -48,
                        53, 54, 55, 56, 49, 50, 51, 52])

    if env.env.include_additional_info:
        obs_per = np.concatenate((obs_per, np.array([58, 57])))
        obs_per = np.concatenate((obs_per, np.array([59])))
        obs_per = np.concatenate((obs_per, np.array([63, 64, -65, 60, 61, -62])))
        obs_per = np.concatenate((obs_per, np.array([66, 67, -68])))
        obs_per = np.concatenate((obs_per, np.array([72, 73, -74, 69, 70, -71])))
        obs_per = np.concatenate((obs_per, np.array([75, 76, -77])))
        obs_per = np.concatenate((obs_per, np.array([78, 79, -80])))
        assert env.env.obs_dim == (57 + 3 + 3 * 6 + 3)
        assert env.env.act_dim == 97            # change action/state permutation if change action/state in env

    def policy_fn(name, ob_space, ac_space):
        old_act_permute = [-86, 87, -88, 93, 94, 95, 96, 89, 90, 91, 92]
        mus_act_l = np.arange(43)
        mus_act_r = mus_act_l + 43
        mus_act_l[0] = 0.001
        act_permute = np.concatenate([mus_act_r, mus_act_l, old_act_permute])
        if env.env.env.state_self_standardize:
            return mlp_mirror_norms_policy.MlpMirrorNormsPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                                hid_size=args.hsize, num_hid_layers=args.layers,
                                                                gmm_comp=1,
                                                                mirror_loss=True,
                                                                observation_permutation=obs_per,
                                                                action_permutation=act_permute)
        else:
            return mlp_mirror_policy.MlpMirrorPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                                     hid_size=args.hsize, num_hid_layers=args.layers, gmm_comp=1,
                                                     mirror_loss=True,
                                                     observation_permutation=obs_per,
                                                     action_permutation=act_permute)

    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    gym.logger.setLevel(logging.WARN)

    joblib.dump(str(env.env.env.__dict__), logger.get_dir() + '/env_specs.pkl', compress=True)
    with open(logger.get_dir() + '/env_specs.txt', 'w') as f:
        pprint.pprint(env.env.env.__dict__, f)
    f.close()
    shutil.copyfile(env.env.env.model_file_name, logger.get_dir() + '/using_model.skel')

    cur_sym_loss = 3.0
    iter_num = 0
    previous_params = None
    # previous_params = joblib.load('')
    reward_threshold = None
    rollout_length_threshold = None
    pposgd_mirror.learn(env, policy_fn,
                        max_timesteps=num_timesteps,
                        timesteps_per_batch=int(2000),
                        clip_param=args.clip, entcoeff=0.0,
                        optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                        gamma=0.99, lam=0.95, schedule='linear',
                        callback=callback,
                        sym_loss_weight=cur_sym_loss,
                        init_policy_params=previous_params,
                        reward_drop_bound=None,
                        rollout_length_threshold=rollout_length_threshold,
                        policy_scope='pi' + str(iter_num),
                        return_threshold=reward_threshold,
                        )

    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartHumanWalkerMD-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--hsize', type=int, default=80)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--clip', type=float, default=0.2)

    parser.add_argument('--HW_final_tar_v', help='final target velocity', type=float, default=1.7)
    parser.add_argument('--HW_tar_acc_time', help='time to acc to final target velocity', type=float, default=1.1)
    parser.add_argument('--HW_energy_weight', help='energy pen weight', type=float, default=0.3)
    parser.add_argument('--HW_alive_bonus_rew', help='alive bonus weight', type=float, default=5.0)
    parser.add_argument('--HW_vel_reward_weight', help='velocity pen weight', type=float, default=9.0)
    parser.add_argument('--HW_side_devia_weight', help='side deviation pen weight', type=float, default=1.5)
    parser.add_argument('--HW_jl_pen_weight', help='joint limit pen weight', type=float, default=0.7)
    parser.add_argument('--HW_alive_pen', help='alive pen weight', type=float, default=0.0)

    args = parser.parse_args()
    logger.reset()

    import datetime
    now = datetime.datetime.now()
    stampstring = now.isoformat()

    logdir = 'data/wtoe_MD_20080_ppo_noAssist_adds' + stampstring[:16] + args.env + '_' + str(
        args.seed) + '_' + str(args.hsize) + '-' + str(args.layers) + '_' + str(args.clip)
    for arg in vars(args):
        if arg[:3] == 'HW_':
            logdir += arg[2:6]
            logdir += '_'
            logdir += arg[-3:]
            logdir += str(getattr(args, arg))
    logger.configure(logdir)
    train_mirror(args, num_timesteps=int(2000 * 8 * 1600))


if __name__ == '__main__':
    main()
