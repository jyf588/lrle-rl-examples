__author__ = 'yuwenhao'

import gym
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import sys, os, time, errno

import joblib
import numpy as np

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from gym import wrappers
import tensorflow as tf
from baselines.ppo1 import mlp_policy, mlp_norms_policy
import baselines.common.tf_util as U
import json

import re

# np.random.seed(10)

state_self_standardize = True
hsize = 80
layers = 2
save_render_data = False
render_path = 'render_data/' + 'humanoid_run_box'


def policy_fn(name, ob_space, ac_space):
    if state_self_standardize:
        return mlp_norms_policy.MlpNormsPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                               hid_size=hsize, num_hid_layers=layers, gmm_comp=1)
    else:
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=hsize, num_hid_layers=layers, gmm_comp=1)


def save_one_frame_shape(env, fpath, step):
    robo_skel = env.env.robot_skeleton
    data = []
    for b in robo_skel.bodynodes:
        if len(b.shapenodes) == 0:
            continue
        if 'cover' in b.name:
            continue
        shape_transform = b.T.dot(b.shapenodes[0].relative_transform()).tolist()
        # pos = trans.translation_from_matrix(shape_transform)
        # rot = trans.euler_from_matrix(shape_transform)
        shape_class = str(type(b.shapenodes[0].shape))
        if 'Mesh' in shape_class:
            stype = 'Mesh'
            path = b.shapenodes[0].shape.path()
            scale = b.shapenodes[0].shape.scale().tolist()
            sub_data = [path, scale]
        elif 'Box' in shape_class:
            stype = 'Box'
            sub_data = b.shapenodes[0].shape.size().tolist()
        elif 'Ellipsoid' in shape_class:
            stype = 'Ellipsoid'
            sub_data = b.shapenodes[0].shape.size().tolist()
        elif 'MultiSphere' in shape_class:
            stype = 'MultiSphere'
            sub_data = b.shapenodes[0].shape.spheres()
            for s in range(len(sub_data)):
                sub_data[s]['pos'] = sub_data[s]['pos'].tolist()

        data.append([stype, b.name, shape_transform, sub_data])
    file = fpath + '/frame_' + str(step) + '.txt'
    json.dump(data, open(file, 'w'))


if __name__ == '__main__':

    sess = tf.InteractiveSession()

    interpolate = 0
    prev_state = None
    render_step = 0

    try:
        os.makedirs(render_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if len(sys.argv) > 1:
        env = gym.make(sys.argv[1])
    else:
        env = gym.make('DartWalker3d-v1')

    if hasattr(env.env, 'disableViewer'):
        env.env.disableViewer = False
        '''if hasattr(env.env, 'resample_MP'):
        env.env.resample_MP = False'''

    record = False
    if len(sys.argv) > 3:
        record = int(sys.argv[3]) == 1
    if record:
        env_wrapper = wrappers.Monitor(env, 'data/videos/', force=True)
    else:
        env_wrapper = env

    U.ALREADY_INITIALIZED = set()
    U.ALREADY_INITIALIZED.update(set(tf.global_variables()))

    # env.env._seed(27)

    policy = None
    if len(sys.argv) > 2:
        policy_params = joblib.load(sys.argv[2])
        ob_space = env.observation_space
        ac_space = env.action_space
        policy = policy_fn("pi", ob_space, ac_space)

        U.initialize()

        cur_scope = policy.get_variables()[0].name[0:policy.get_variables()[0].name.find('/')]
        orig_scope = list(policy_params.keys())[0][0:list(policy_params.keys())[0].find('/')]
        vars = policy.get_variables()

        for i in range(len(policy.get_variables())):
            assign_op = policy.get_variables()[i].assign(
                policy_params[policy.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
            sess.run(assign_op)

    print('===================')

    final_v_strs = re.findall('fin_r_v\d+\.\d+', sys.argv[2])
    if final_v_strs:
        env.env.final_tar_v = float(final_v_strs[0][7:])
    tar_time_strs = re.findall('tar_ime\d+\.\d+', sys.argv[2])
    if tar_time_strs:
        env.env.tar_acc_time = float(tar_time_strs[0][7:])
    lr_string = re.findall('mus_mitTrue', sys.argv[2])
    le_string = re.findall('mus_ostTrue', sys.argv[2])
    env.env.muscle_add_tor_limit = lr_string
    env.env.muscle_add_energy_cost = le_string
    env.env.init_params(None)

    o = env_wrapper.reset()

    rew = 0

    actions = []

    traj = 1
    ct = 0
    vel_rew = []
    action_pen = []
    deviation_pen = []
    rew_seq = []

    x_vel = []
    x_vel2 = []
    x_vel3 = []
    # foot_contacts = []

    avg_vels = []
    meta_costs = []
    d = False
    step = 0
    save_qs = []
    save_dqs = []

    while ct < traj:
        if policy is not None:
            ac, vpred = policy.act(step < 0, o)
            # ac, vpred = policy.act(True, o)
            act = ac
        else:
            act = env.action_space.sample()

        # time.sleep(0.1)
        actions.append(act)

        '''if env_wrapper.env.env.t > 3.0 and env_wrapper.env.env.t < 6.0:
            env_wrapper.env.env.robot_skeleton.bodynode('head').add_ext_force(np.array([-200, 0, 0]))'''
        o, r, d, env_info = env_wrapper.step(act)

        if 'action_pen' in env_info:
            action_pen.append(env_info['action_pen'])
        if 'vel_rew' in env_info:
            vel_rew.append(env_info['vel_rew'])
        rew_seq.append(r)
        if 'deviation_pen' in env_info:
            deviation_pen.append(env_info['deviation_pen'])
        if 'avg_vel' in env_info:
            avg_vels.append(env_info['avg_vel'])
        if 'meta_cost' in env_info:
            meta_costs.append(env_info['meta_cost'])

        rew += r
        # foot_contacts.append(o[57:59])

        env_wrapper.render()
        step += 1

        # time.sleep(0.1)
        # if len(o) > 25:
        x_vel.append(env.env.robot_skeleton.dq[0])
        x_vel2.append(env.env.vel)
        # print(env.env.vel)
        x_vel3.append(env.env.target_vel)

        save_qs.append(env.env.robot_skeleton.q)
        save_dqs.append(env.env.robot_skeleton.dq)

        if save_render_data:
            cur_state = env.env.state_vector()
            if prev_state is not None and interpolate > 0:
                for it in range(interpolate):
                    int_state = (it + 1) * 1.0 / (interpolate + 1) * prev_state + (
                                1 - (it + 1) * 1.0 / (interpolate + 1)) * cur_state
                    env.env.set_state_vector(int_state)
                    save_one_frame_shape(env, render_path, render_step)
                    render_step += 1
            env.env.set_state_vector(cur_state)
            save_one_frame_shape(env, render_path, render_step)
            render_step += 1
            prev_state = env.env.state_vector()

        if d:
            step = 0
            ct += 1
            print('reward: ', rew)
            o = env_wrapper.reset()
            # break
    print('avg rew ', rew / traj)
    print('total energy penalty: ', np.sum(action_pen) / traj)
    print('total vel rew: ', np.sum(vel_rew) / traj)

    if sys.argv[1] == 'DartWalker3d-v1' or sys.argv[1] == 'DartWalker3dSPD-v1':
        rendergroup = [[0, 1, 2], [3, 4, 5, 9, 10, 11], [6, 12], [7, 8, 12, 13]]
        for rg in rendergroup:
            plt.figure()
            for i in rg:
                plt.plot(np.array(actions)[:, i])
    if sys.argv[1] == 'DartHumanWalker-v1':
        rendergroup = [[0,1,2, 6,7,8], [3,9], [4,5,10,11], [12,13,14], [15,16,7,18]]
        # rendergroup = [[0, 6, ], [3, 9], [4, 10], [12, 13, 14], [15, 16, 7, 18]]
        titles = ['thigh', 'knee', 'foot', 'waist', 'arm']
        for i, rg in enumerate(rendergroup):
            plt.figure()
            plt.title(titles[i])
            for i in rg:
                plt.plot(np.array(actions)[:, i])
    if sys.argv[1] == 'DartHumanWalker-v2':
        rendergroup = [[0, 5, ], [3, 8], [4, 9], [10, 11, 12], [13, 17]]
        titles = ['thigh', 'knee', 'foot', 'torso', 'shoulder']
        for i, rg in enumerate(rendergroup):
            plt.figure()
            plt.title(titles[i])
            for i in rg:
                plt.plot(np.array(actions)[:, i])
    if sys.argv[1] == 'DartDogRobot-v1':
        rendergroup = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        titles = ['rear right leg', 'rear left leg', 'front right leg', 'front left leg']
        for i, rg in enumerate(rendergroup):
            plt.figure()
            plt.title(titles[i])
            for i in rg:
                plt.plot(np.array(actions)[:, i])
    if sys.argv[1] == 'DartHexapod-v1':
        rendergroup = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17]]
        titles = ['hind legs', 'middle legs', 'front legs']
        for i, rg in enumerate(rendergroup):
            plt.figure()
            plt.title(titles[i])
            for i in rg:
                plt.plot(np.array(actions)[:, i])
    plt.figure()
    plt.title('rewards')
    plt.plot(rew_seq, label='total rew')
    plt.plot(action_pen, label='action pen')
    plt.plot(vel_rew, label='vel rew')
    plt.plot(deviation_pen, label='dev pen')
    plt.legend()
    plt.figure()
    plt.title('x vel')
    plt.plot(x_vel)
    plt.plot(x_vel2)
    plt.plot(x_vel3)

    # foot_contacts = np.array(foot_contacts)
    # plt.figure()
    # plt.title('foot contacts')
    # plt.plot(foot_contacts[:, 0])
    # plt.plot(foot_contacts[:, 1])

    plt.figure()
    plt.title('average velocity')
    plt.plot(avg_vels)
    plt.figure()
    plt.title('meta cost')
    plt.plot(meta_costs)

    print('total vel rewrads ', np.sum(vel_rew))
    print('total action rewards ', np.sum(action_pen))

    plt.figure()
    plt.title("hip angle")
    plt.plot(np.array(save_qs)[:, 6])
    plt.plot(np.array(save_qs)[:, 12])

    plt.figure()
    plt.title("ankle q")
    plt.plot(np.array(save_qs)[:, 10])
    plt.plot(np.array(save_qs)[:, 16])

    # ################ save average action signals #################
    # avg_action = np.mean(np.abs(actions), axis=1)
    # np.savetxt('data/force_data/action_mean.txt', avg_action)
    # np.savetxt('data/force_data/action_std.txt', np.std(np.abs(actions), axis=1))

    plt.show()
