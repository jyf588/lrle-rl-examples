__author__ = 'yifengjiang'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import os
import json

import pydart2 as pydart

from keras.models import load_model


# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


def load_model_weights(pathname):
    w = []
    b = []
    from os import path
    if pathname.startswith("/"):
        fullpath = pathname
    else:
        fullpath = os.path.join(os.path.dirname(__file__), "assets", pathname)
    if not path.exists(fullpath):
        raise IOError("File %s does not exist" % fullpath)
    model = load_model(fullpath)
    # NOTE! assume no act final layer
    for i in range(0, len(model.layers)):
        # workaround for dropout layers
        ws = model.layers[i].get_weights()
        if len(ws) > 0:
            w.append(ws[0])
            b.append(ws[1])

    return w, b


def neural_net_regress_elu(x, wmats, bmats, input_mult=None, output_mult=None):
    if input_mult is None:
        input_mult = np.ones(wmats[0].shape[0])
    if output_mult is None:
        output_mult = np.ones(bmats[-1].shape[0])  # B is 1D anyways

    x = x * input_mult
    wxb = x[np.newaxis, :]
    for i in range(0, len(wmats)):
        wxb = wxb.dot(wmats[i]) + bmats[i]
        if i != (len(wmats) - 1):
            wxb = np.exp(wxb * (wxb <= 0)) - 1 + wxb * (wxb > 0)

    return wxb[0, :] * output_mult


class DartHumanWalkerToeEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        # action/state related params
        self.control_bounds = np.array([[1.0] * 21, [-1.0] * 21])

        self.action_scale = np.array([200.0, 80, 80, 250, 250, 200, 80, 80, 250, 250,
                                      150, 150, 100,
                                      100, 30, 15, 30, 100, 30, 15, 30])
        #
        # self.action_scale = np.array([160.0, 60, 60, 160, 160, 160.0, 60, 60, 160, 160,
        #                               150, 150, 100,
        #                               100, 30, 15, 30, 100, 30, 15, 30])

        self.previous_control = None
        self.constrain_dcontrol = 1.0

        self.state_self_standardize = True
        self.obs_dim = 57

        self.include_additional_info = True
        self.joints_to_include = ['j_heel_left', 'j_heel_right', 'j_head']
        self.contact_info = np.array([0, 0])
        if self.include_additional_info:
            self.obs_dim += len(self.contact_info)
            self.obs_dim += 1  # target vel
            self.obs_dim += len(self.joints_to_include) * 3 * 2  # each joint relative-pos and vel is in xyz
            self.obs_dim += 3  # com vel

        # init env
        self.metadata = {
            "render.modes": ['human', 'rgb_array'],
            "video.frames_per_second": 33
        }
        self.model_file_name = 'kima/kima_human_opensim_wtoe.skel'
        self.model_file_name = os.path.join(os.path.dirname(__file__), "assets", self.model_file_name)
        if not os.path.exists(self.model_file_name):
            raise IOError("File %s does not exist" % self.model_file_name)
        dart_env.DartEnv.__init__(self, self.model_file_name, 15, self.obs_dim, self.control_bounds,
                                  disableViewer=True, dt=0.002)
        self.init_env()

        self.init_height = self.robot_skeleton.bodynode('head').C[1]
        self.sim_dt = self.dt / self.frame_skip
        self.t = 0
        self.cur_step = 0
        self.ltoe = self.robot_skeleton.joint('j_toe_left').dofs[0]
        self.rtoe = self.robot_skeleton.joint('j_toe_right').dofs[0]

        # muscle model init
        self.muscle_add_tor_limit = False  # set outside
        self.modelR_path = 'neuralnets/InOutR_2_4M_3D_q-0915&-0606&-0606&-2101&-0909_dq555&10&15_tau250&80&80&250&250_2392_may25_elu.h5'
        self.RWmats = []
        self.RBmats = []

        self.muscle_add_energy_cost = False  # set outside
        self.modelE_path = 'neuralnets/InOutRE_ValidR4E_3D_q-0915&-0606&-0606&-2101&-0909_dq555&10&15_tau250&80&80&250&250_2392_may25_mse_fix_simple_as_volumn.h5'
        self.EWmats = []
        self.EBmats = []

        # # TODO: for DEBUG only
        # fullpath = os.path.join(os.path.dirname(__file__), "assets", self.modelR_path)
        # self.modelR = load_model(fullpath)
        # fullpath = os.path.join(os.path.dirname(__file__), "assets", self.modelE_path)
        # self.modelE = load_model(fullpath)
        # self.ankle_taus = []
        # self.ankle_dq = []
        # self.ankle_q = []
        # self.dy = []
        # self.residues = []
        # self.energys = []

        # muscle model related params
        self.input_mult = np.ones(15)
        self.input_mult[5:9] /= 2.5
        self.input_mult[9:10] /= 7.5
        self.input_mult[10:15] /= 100.0
        # dq is trained in 555,10,15
        self.dq_clip = np.array([5.0, 5, 5, 10, 15])
        self.output_mult_E = np.array([10.0])
        self.dq_cache_l = []
        self.dq_cache_r = []
        # need to calc residue when using meta energy based on actual(not proposed) tau
        self.total_residue = None
        self.meta_cost = None
        self.up_energy = 700.0  # change this scaling factor if using a different metabolic to train E

        # reward related
        self.energy_weight = 0.3  # set outside
        self.alive_bonus_rew = 9.0  # set outside
        self.vel_reward_weight = 7.0  # set outside
        self.side_devia_weight = 1.5  # set outside
        self.jl_pen_weight = 0.7  # set outside
        self.alive_pen = 0.0  # set outside

        self.residue_pen_weight = 3.0  # set outside

        self.final_tar_v = 1.4  # set outside
        self.tar_acc_time = 1.7  # set outside
        self.target_vel = None
        self.vel = None
        self.vel_cache = []

        self.save_render_data = False
        self.render_path = 'render_data/' + 'humanoid_walkORrun_new'
        self.render_step = 0
        import errno
        try:
            os.makedirs(self.render_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        utils.EzPickle.__init__(self)

    def init_env(self):
        self.robot_skeleton.set_self_collision_check(True)
        self.robot_skeleton.set_adjacent_body_check(False)

        for i in range(0, len(self.dart_world.skeletons[0].bodynodes)):
            self.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(20)
        for i in range(0, len(self.dart_world.skeletons[1].bodynodes)):
            self.dart_world.skeletons[1].bodynodes[i].set_friction_coeff(20)

        # self.dart_world.set_collision_detector(3)

        for bn in self.robot_skeleton.bodynodes:
            if len(bn.shapenodes) > 0:
                if hasattr(bn.shapenodes[0].shape, 'size'):
                    shapesize = bn.shapenodes[0].shape.size()
                    print('density of ', bn.name, ' is ', bn.mass() / np.prod(shapesize))
        print('Total mass: ', self.robot_skeleton.mass())

    def init_params(self, args):
        if args is not None:
            for arg in vars(args):
                if arg[:3] == 'HW_' and hasattr(self, arg[3:]):
                    setattr(self, arg[3:], getattr(args, arg))

        if self.muscle_add_tor_limit:
            self.RWmats, self.RBmats = load_model_weights(self.modelR_path)
            if self.muscle_add_energy_cost:
                self.EWmats, self.EBmats = load_model_weights(self.modelE_path)
        else:   # make range smaller for box, otherwise wont train at all, could try other combinations
            self.action_scale = np.array([160.0, 60, 60, 160, 160, 160.0, 60, 60, 160, 160, 
                                            150, 150, 100,
                                            100, 30, 15, 30, 100, 30, 15, 30])

        # print(self.__dict__)

    def save_one_frame_shape(self, fpath, step):
        robo_skel = self.robot_skeleton
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

    def do_simulation(self, tau, n_frames):
        self.total_residue = 0.0
        self.meta_cost = 0.0
        residue_l = np.zeros([5])
        residue_r = np.zeros([5])
        for frame in range(n_frames):
            tau_revised = np.copy(tau)

            self.dq_cache_l.append(self.robot_skeleton.dq[6:11])
            self.dq_cache_r.append(self.robot_skeleton.dq[12:17])

            if len(self.dq_cache_l) > 5:
                self.dq_cache_l.pop(0)
                self.dq_cache_r.pop(0)

            if frame % 5 == 0:
                if self.save_render_data:
                    self.save_one_frame_shape(self.render_path, self.render_step)
                    self.render_step += 1

            if self.muscle_add_tor_limit:
                if frame % 5 == 0:  # Assume residue will not change in this 0.01s
                    dq_in_l = np.clip(np.mean(self.dq_cache_l, axis=0), -self.dq_clip, self.dq_clip)
                    cr_in_l = np.concatenate([self.robot_skeleton.q[6:11], dq_in_l, tau[6:11]])
                    residue_l = neural_net_regress_elu(cr_in_l, self.RWmats, self.RBmats, input_mult=self.input_mult)
                    self.total_residue += np.linalg.norm(residue_l - np.clip(residue_l, -10.0, 10.0))

                    dq_in_r = np.clip(np.mean(self.dq_cache_r, axis=0), -self.dq_clip, self.dq_clip)
                    cr_in_r = np.concatenate([self.robot_skeleton.q[12:17], dq_in_r, tau[12:17]])
                    residue_r = neural_net_regress_elu(cr_in_r, self.RWmats, self.RBmats, input_mult=self.input_mult)
                    self.total_residue += np.linalg.norm(residue_r - np.clip(residue_r, -10.0, 10.0))

                    if self.muscle_add_energy_cost:
                        cr_in_l[10:15] -= residue_l
                        energy_rate = neural_net_regress_elu(cr_in_l, self.EWmats, self.EBmats,
                                                             input_mult=self.input_mult,
                                                             output_mult=self.output_mult_E)
                        self.meta_cost += np.maximum(energy_rate[0], 0.0)
                        cr_in_r[10:15] -= residue_r
                        energy_rate = neural_net_regress_elu(cr_in_r, self.EWmats, self.EBmats,
                                                             input_mult=self.input_mult,
                                                             output_mult=self.output_mult_E)
                        self.meta_cost += np.maximum(energy_rate[0], 0.0)
                        # print("E:")
                        # print(x)
                        # print(energy_rate[0])
                        # x = x * self.input_mult
                        # print(self.modelE.predict(x[np.newaxis, :], batch_size=1))
                tau_revised[6:11] -= residue_l
                tau_revised[12:17] -= residue_r

                # self.residues.append(np.linalg.norm(residue_r* (np.abs(residue_r) > 10.0)))
                # print("RR:")
                # print(cr_in_r)
                # print(residue_r)
                # cr_in_r = cr_in_r * self.input_mult
                # print(self.modelR.predict(cr_in_r[np.newaxis, :], batch_size=1))
            # print(tau_revised)
            self.robot_skeleton.set_forces(tau_revised)

            self.dart_world.check_collision()   # TODO: ideally step() should not call collide again, but no api
            self.ltoe.set_coulomb_friction(10.0)
            self.rtoe.set_coulomb_friction(10.0)
            cbs = self.dart_world.collision_result.contacted_bodies
            if self.robot_skeleton.bodynode('l-foot') not in cbs and \
               self.robot_skeleton.bodynode('l-toe') not in cbs:
                self.ltoe.set_coulomb_friction(0.0)
            if self.robot_skeleton.bodynode('r-foot') not in cbs and \
               self.robot_skeleton.bodynode('r-toe') not in cbs:
                self.rtoe.set_coulomb_friction(0.0)

            # self.ankle_taus.append(tau_revised[10])
            # self.ankle_dq.append(self.robot_skeleton.dq[10])
            # self.ankle_q.append(self.robot_skeleton.q[10])
            # self.dy.append(self.robot_skeleton.dq[1])
            # print(self.robot_skeleton.q[17])
            self.dart_world.step()
            if self.is_broke_sim():
                break

        self.total_residue = self.total_residue / 3 / 160.0 * self.residue_pen_weight  # (frame skip15/5=3, 160N/m scaling)
        self.meta_cost = self.meta_cost / 6 * 10 / self.up_energy   # (frame skip15/3*2=6, 10 DoFs for 2legs)

    def advance(self, clamped_control):
        tau = np.zeros(self.robot_skeleton.ndofs)
        ctrl_tau = clamped_control * self.action_scale
        ctrl_tau = np.concatenate((ctrl_tau[:5], [0], ctrl_tau[5:10], [0], ctrl_tau[10:]))

        tau[6:] = ctrl_tau
        self.do_simulation(tau, self.frame_skip)

    def clamp_act(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
            if self.previous_control is not None:
                if clamped_control[i] > self.previous_control[i] + self.constrain_dcontrol:
                    clamped_control[i] = self.previous_control[i] + self.constrain_dcontrol
                elif clamped_control[i] < self.previous_control[i] - self.constrain_dcontrol:
                    clamped_control[i] = self.previous_control[i] - self.constrain_dcontrol
        return clamped_control

    def calc_joint_limits_pen(self):
        joints = ['j_thigh_left', 'j_thigh_right', 'j_heel_left', 'j_heel_right',
                  'j_bicep_left', 'j_bicep_right', 'j_forearm_left', 'j_forearm_right',
                  'j_abdomen', 'j_spine']
        starting_dof_inds = [6, 12, 10, 16, 21, 25, 24, 28, 18, 20]

        joint_limit_pen = 0
        ind = 0
        for j_name in joints:
            j = self.robot_skeleton.joint(j_name)
            dof_ind = starting_dof_inds[ind]
            for i in range(j.num_dofs()):
                joint_limit_pen += np.minimum(np.exp((j.position_lower_limit(i) - self.robot_skeleton.q[dof_ind + i])
                                                     / 0.04), 1.0)
                joint_limit_pen += np.minimum(np.exp((self.robot_skeleton.q[dof_ind + i] - j.position_upper_limit(i))
                                                     / 0.04), 1.0)
            ind += 1
        return self.jl_pen_weight * joint_limit_pen

    def calc_head_pen(self):
        head_vel = self.robot_skeleton.bodynode('head').dC[1:]  # exclude x vel
        return self.side_devia_weight * np.sum(np.abs(head_vel)) / 5.0

    def is_broke_sim(self):
        s = self.state_vector()
        return not (np.isfinite(s).all() and (np.abs(s[3:]) < 10).all())

    def _step(self, a):
        posbefore = self.robot_skeleton.com()[0]

        clamped_control = self.clamp_act(a)

        self.advance(np.copy(clamped_control))

        self.t += self.dt
        self.cur_step += 1

        posafter = self.robot_skeleton.com()[0]

        height = self.robot_skeleton.bodynode('head').com()[1]
        side_deviation = self.robot_skeleton.bodynode('head').com()[2]

        contacts = self.dart_world.collision_result.contacts
        self.contact_info = np.array([0, 0])
        for contact in contacts:
            if contact.skel_id1 + contact.skel_id2 == 1:
                # not self-colliding
                if contact.bodynode1 == self.robot_skeleton.bodynode('l-foot') or \
                        contact.bodynode2 == self.robot_skeleton.bodynode('l-foot') or \
                        contact.bodynode1 == self.robot_skeleton.bodynode('l-toe') or \
                        contact.bodynode2 == self.robot_skeleton.bodynode('l-toe'):
                    self.contact_info[0] = 1
                if contact.bodynode1 == self.robot_skeleton.bodynode('r-foot') or \
                        contact.bodynode2 == self.robot_skeleton.bodynode('r-foot') or \
                        contact.bodynode1 == self.robot_skeleton.bodynode('r-toe') or \
                        contact.bodynode2 == self.robot_skeleton.bodynode('r-toe'):
                    self.contact_info[1] = 1

        hp = [-0.3, -0.06]
        abp = [0.5, 1]
        alive_bonus = np.interp(height - self.init_height, hp, abp) * self.alive_bonus_rew
        # during running, should not encourage stay alive with some low speed
        # i.e. reward should not be improved by simply live longer
        alive_bonus -= self.alive_pen

        self.vel = (posafter - posbefore) / self.dt
        self.vel_cache.append(self.vel)
        if len(self.vel_cache) > int(1.0 / self.dt):
            self.vel_cache.pop(0)

        # smoothly increase the target velocity
        self.target_vel = (np.min([self.t, self.tar_acc_time]) / self.tar_acc_time) * self.final_tar_v
        # for numerical stability (/0)
        stable_target_vel = np.maximum(self.target_vel, 0.1)
        # speed over target_vel will not be rewarded
        vel_reward = self.vel_reward_weight * np.minimum(self.vel, stable_target_vel) / stable_target_vel

        a_leg = np.copy(a[:10])
        a_non_leg = np.copy(a[10:])
        action_pen = 0.0
        if self.muscle_add_tor_limit and self.muscle_add_energy_cost:
            action_pen += self.energy_weight * self.meta_cost
        else:
            action_pen += self.energy_weight * np.abs(a_leg).sum()
        action_pen += np.minimum(self.energy_weight, 0.4) * np.abs(a_non_leg).sum()

        deviation_pen = self.side_devia_weight * abs(side_deviation)

        reward = - action_pen - deviation_pen - self.calc_head_pen() - self.calc_joint_limits_pen()
        if self.muscle_add_tor_limit:
            reward -= 2 * np.sqrt(self.total_residue + 1) - 2
        reward += vel_reward
        reward += alive_bonus

        # print('ab', alive_bonus)
        # print('vel',self.vel)
        # print(self.robot_skeleton.dC[0])
        # print(vel_reward)

        self.previous_control = clamped_control

        ob = self._get_obs()

        broke_sim = self.is_broke_sim()
        alive = (height - self.init_height > -0.3) and (height - self.init_height < 1.0) and (
                np.abs(side_deviation) < 0.9) and (not broke_sim)
        done = not alive

        # if done:
        #     # print(not no_progress)
        #     # print(s)
        #     print(self.robot_skeleton.dq[6:])
        #     # print(ang_cos_fwd)
        #     print((np.abs(s[3:]) < 10).all())
        #     print((height - self.init_height > -0.3))

        # if self.cur_step == 499:
        # plt.figure()
        # plt.title('residues')
        # plt.plot(self.residues, label='residues')
        # plt.figure()
        # plt.title('energies')
        # plt.plot(self.energys)
        # plt.figure()
        # plt.title('energies')
        # plt.hist(self.energys)
        #     input("press enter to continue...")
        #     plt.figure()
        #     plt.title('q')
        #     plt.plot(self.ankle_q, label='q')
        #     plt.figure()
        #     plt.title('dq')
        #     plt.plot(self.ankle_dq, label='dq')
        #     plt.figure()
        #     plt.title('tau')
        #     plt.plot(self.ankle_taus, label='tau')
        #     plt.figure()
        #     plt.title('dy')
        #     plt.plot(self.dy, label='dy')
        #     plt.show()

        '''work = np.sum(np.abs(clamped_control * self.action_scale * self.robot_skeleton.dq[6:])) * self.dt
        self.total_work += work
        cot = self.total_work / (self.robot_skeleton.mass() * 9.81 * self.robot_skeleton.C[0])
        print(cot)'''

        return ob, reward, done, {'broke_sim': broke_sim, 'vel_rew': vel_reward, 'action_pen': action_pen,
                                  'deviation_pen': deviation_pen, 'done_return': done,
                                  'dyn_model_id': 0, 'state_index': 0, 'com': self.robot_skeleton.com(),
                                  'meta_cost': self.meta_cost, 'avg_vel': np.mean(self.vel_cache)
                                  }

    def _get_obs(self):
        state = np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq[:6] / 2.0,
            self.robot_skeleton.dq[6:] / 10.0,
        ])

        if self.include_additional_info:
            state = np.concatenate([state, self.contact_info])
            state = np.concatenate([state, [self.target_vel / self.final_tar_v]])

            joint_pos = np.array([])
            joint_vel = np.array([])
            for j_name in self.joints_to_include:
                joint = self.robot_skeleton.joint(j_name)
                joint_pos = np.concatenate([joint_pos, joint.child_bodynode.C - self.robot_skeleton.q[:3]])
                joint_vel = np.concatenate([joint_vel, joint.child_bodynode.dC / 5.0])
            state = np.concatenate([state, joint_pos, joint_vel])

            state = np.concatenate([state, self.robot_skeleton.dC / 2.0])

        return state

    def reset_model(self):
        # print('resetttt')
        self.dart_world.reset()

        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-0.01, high=0.01, size=self.robot_skeleton.ndofs)
        qpos[[0, 1, 2]] = 0
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-0.05, high=0.05, size=self.robot_skeleton.ndofs)

        self.set_state(qpos, qvel)

        self.t = 0.0
        self.cur_step = 0

        self.contact_info = np.array([0, 0])

        self.previous_control = None

        self.target_vel = 0.0
        self.vel = 0.0
        self.vel_cache = []
        self.dq_cache_l = []
        self.dq_cache_r = []

        self.total_residue = None
        self.meta_cost = None

        self.render_step = 0

        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            # self.track_skeleton_id = 0
            self._get_viewer().scene.tb.trans[2] = -5.5
