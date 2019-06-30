__author__ = 'yifengjiang'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import os

import pydart2 as pydart

from keras.models import load_model

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


def load_model_weights(pathname):
    W = []
    B = []
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
            W.append(ws[0])
            B.append(ws[1])

    return W, B


def neural_net_regress_elu(x, Wmats, Bmats, input_mult=None, output_mult=None):
    if input_mult is None:
        input_mult = np.ones(Wmats[0].shape[0])
    if output_mult is None:
        output_mult = np.ones(Bmats[-1].shape[0])   # B is 1D anyways

    x = x * input_mult
    wxb = x[np.newaxis, :]
    for i in range(0, len(Wmats)):
        wxb = wxb.dot(Wmats[i]) + Bmats[i]
        if i != (len(Wmats) - 1):
            wxb = np.exp(wxb*(wxb<=0))-1 + wxb*(wxb>0)

    return wxb[0, :]*output_mult


class DartHumanWalkerEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):

        self.metadata = {
            "render.modes": ['human', 'rgb_array'],
            "video.frames_per_second": 33
        }

        # action related params
        self.control_bounds = np.array([[1.0] * 23, [-1.0] * 23])

        # self.action_scale = np.array([160.0, 60, 60, 100, 80, 60, 160, 60, 60, 100, 80, 60,
        #                               150, 150, 100,
        #                               100, 15, 15, 30, 100,15,15, 30])  # default tor lim without NN
        self.action_scale = np.array([200.0, 60, 60, 200, 200, 60, 200, 60, 60, 200, 200, 60,
                                      150, 150, 100,
                                      100, 15, 15, 30, 100, 15, 15, 30])
        # # TODO: changed in rot to 0.6, change maybe box limit form 60 to 100
        # self.action_scale = np.array([200.0, 100, 100, 200, 200, 60, 200, 100, 100, 200, 200, 60,
        #                               150, 150, 100,
        #                               100, 15, 15, 30, 100, 15, 15, 30])

        obs_dim = 57

        self.t = 0
        self.cur_step = 0

        # muscle model init
        self.muscle_add_tor_limit = False   # set outside   #TODO
        # self.modelR_path = 'neuralnets/InOutR_1_5M_3D_-0915&-0606&-0606&-2101&-0909_dq555&10&15_tau200&100&100&200&200_2392_dec31_elu_2.h5'
        self.modelR_path = 'neuralnets/InOutR_1Maddvalid600k_3D_q-0920&-0606&-0602&-2501&-1010_dq555&10&15_tau200&60&60&200&200_2392_dec8_elu.h5'
        self.RWmats = []
        self.RBmats = []

        self.muscle_add_energy_cost = False # set outside
        # self.modelE_path = 'neuralnets/InOutE_Valid_3D_q-0920&-0606&-0602&-2501&-1010_dq555&10&15_tau200&60&60&200&200_2392_dec8_18012864elu_mse_4%.h5'
        # self.modelE_path = 'neuralnets/InOutE_ValidInv_3D_q-0915&-0606&-0602&-2101&-0909_dq555&10&15_tau200&100&100&200&200_2392_dec26_512360128elu_mse_fix_10%.h5'
        # self.modelE_path = 'neuralnets/InOutE_ValidInv_3D_q-0915&-0606&-0602&-2101&-0909_dq555&10&15_tau200&100&100&200&200_2392_dec29_512360128elu_mse_fix_simple.h5'
        # self.modelE_path = 'neuralnets/InOutE_Valid_3D_q-0915&-0606&-0602&-2101&-0909_dq555&10&15_tau200&100&100&200&200_2392_dec29_18012864lu_mse_fix_simple.h5'
        self.modelE_path = 'neuralnets/InOutE_Valid_3D_q-0915&-0606&-0606&-2101&-0909_dq555&10&15_tau200&100&100&200&200_2392_dec31_18012864lu_mse_fix_simple_as.h5'
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

        self.add_flex_noise_q = True    # set outside

        # muscle model related params
        self.input_mult = np.ones(15)
        self.input_mult[5:9] /= 2.5
        self.input_mult[9:10] /= 7.5
        self.input_mult[10:15] /= 100.0
        # dq is trained in 555,10,15
        self.dq_clip = np.array([5.0, 5, 5, 10, 15])
        self.output_mult_E = np.array([100.0])

        # reward related
        self.energy_weight = 0.3  # set outside
        self.alive_bonus_rew = 9.0  # set outside
        self.vel_reward_weight = 4.5  # set outside
        self.side_devia_weight = 3.0   # set outside
        self.rot_pen_weight = 0.3  # set outside
        self.abd_pen_weight = 1.0   # set outside
        self.spl_pen_weight = 0.5   # set outside
        self.angle_pen_weight = 0.8     # set outside

        # need to calc residue when using meta energy based on actual(not proposed) tau
        self.total_residue = 0.0
        self.meta_cost = 0.0
        self.ignore_energy = 2000.0
        self.up_energy = 15000.0
        self.residue_pen = 9.0   #TODO

        self.vel_cache = []
        self.target_vel_cache = []
        self.dq_cache_l = []
        self.dq_cache_r = []
        self.ankle_vel_cache_long = []

        # assistance related
        self.assist_timeout = 0.0  # do not provide pushing assistance after certain time # set outside
        self.assist_schedule = [[0.0, [2000.0, 2000]], [3.0, [1500.0, 1500]], [6.0, [1125.0, 1125]]]
        self.current_pd = None
        self.vel_enforce_kp = None
        self.init_tv = 0.0
        self.final_tv = 1.5  # set outside
        self.tv_endtime = 1.2  # set outside
        self.target_vel = None
        self.tvel_diff_perc = 1.0
        self.push_target = 'pelvis'

        self.previous_control = None
        self.constrain_dcontrol = 1.0

        # additional states
        self.include_additional_info = True
        self.contact_info = np.array([0, 0])
        if self.include_additional_info:
            obs_dim += len(self.contact_info)
            obs_dim += 1  # target vel

        dart_env.DartEnv.__init__(self, 'kima/kima_human_opensim_musclespring.skel', 15, obs_dim, self.control_bounds,
                                  disableViewer=True, dt=0.002)

        self.robot_skeleton.set_self_collision_check(True)
        self.robot_skeleton.set_adjacent_body_check(False)

        self.init_height = self.robot_skeleton.bodynode('head').C[1]

        for i in range(0, len(self.dart_world.skeletons[0].bodynodes)):
            self.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(20)
        for i in range(0, len(self.dart_world.skeletons[1].bodynodes)):
            self.dart_world.skeletons[1].bodynodes[i].set_friction_coeff(20)

        # self.dart_world.set_collision_detector(3)

        self.sim_dt = self.dt / self.frame_skip

        for bn in self.robot_skeleton.bodynodes:
            if len(bn.shapenodes) > 0:
                if hasattr(bn.shapenodes[0].shape, 'size'):
                    shapesize = bn.shapenodes[0].shape.size()
                    print('density of ', bn.name, ' is ', bn.mass() / np.prod(shapesize))
        print('Total mass: ', self.robot_skeleton.mass())

        utils.EzPickle.__init__(self)

    def init_params(self, args):
        if args is not None:
            for arg in vars(args):
                if arg[:3] == 'HW_' and hasattr(self, arg[3:]):
                    setattr(self, arg[3:], getattr(args, arg))

        if self.muscle_add_tor_limit:
            self.RWmats, self.RBmats = load_model_weights(self.modelR_path)
            if self.muscle_add_energy_cost:
                self.EWmats, self.EBmats = load_model_weights(self.modelE_path)

        # print(self.__dict__)

    # only 1d
    def _spd(self, target_q, id, kp, target_dq=None):
        self.Kp = kp
        self.Kd = kp * self.sim_dt
        if target_dq is not None:
            self.Kd = self.Kp
            self.Kp *= 0

        invM = 1.0 / (self.robot_skeleton.M[id][id] + self.Kd * self.sim_dt)
        if target_dq is None:
            p = -self.Kp * (self.robot_skeleton.q[id] + self.robot_skeleton.dq[id] * self.sim_dt - target_q[id])
        else:
            p = 0
        d = -self.Kd * (self.robot_skeleton.dq[id] - target_dq)
        qddot = invM * (-self.robot_skeleton.c[id] + p + d)
        tau = p + d - self.Kd * (qddot) * self.sim_dt

        return tau

    def _bodynode_spd(self, bn, kp, dof, target_vel=None):
        self.Kp = kp
        self.Kd = kp * self.sim_dt
        if target_vel is not None:
            self.Kd = self.Kp
            self.Kp *= 0

        invM = 1.0 / (bn.mass() + self.Kd * self.sim_dt)
        p = -self.Kp * (bn.C[dof] + bn.dC[dof] * self.sim_dt)
        if target_vel is None:
            target_vel = 0.0
        d = -self.Kd * (bn.dC[dof] - target_vel * 1.0)  # compensate for average velocity match
        qddot = invM * (-bn.C[dof] + p + d)
        tau = p + d - self.Kd * (qddot) * self.sim_dt

        return tau

    def do_simulation(self, tau, n_frames):
        self.total_residue = 0.0
        self.meta_cost = 0.0
        residue_l = np.zeros([5])
        residue_r = np.zeros([5])
        for frame in range(n_frames):
            if self.t < self.assist_timeout:
                force = self._bodynode_spd(self.robot_skeleton.bodynode(self.push_target), self.current_pd, 2)
                self.robot_skeleton.bodynode(self.push_target).add_ext_force(np.array([0, 0, force]))

                force = self._bodynode_spd(self.robot_skeleton.bodynode(self.push_target), self.vel_enforce_kp, 0,
                                           self.target_vel * self.tvel_diff_perc)
                self.robot_skeleton.bodynode(self.push_target).add_ext_force(np.array([force, 0, 0]))

            tau_revised = np.copy(tau)

            self.dq_cache_l.append(self.robot_skeleton.dq[6:11])
            self.dq_cache_r.append(self.robot_skeleton.dq[12:17])
            self.ankle_vel_cache_long.append(self.robot_skeleton.dq[[11, 17]])

            if len(self.dq_cache_l) > 3:  # TODO
                self.dq_cache_l.pop(0)
                self.dq_cache_r.pop(0)
            if len(self.ankle_vel_cache_long) > 7:     #TODO
                self.ankle_vel_cache_long.pop(0)

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
                        cr_in_l[9] = np.mean(self.ankle_vel_cache_long, axis=0)[0]      # TODO
                        energy_rate = neural_net_regress_elu(cr_in_l, self.EWmats, self.EBmats, input_mult=self.input_mult,
                                                             output_mult=self.output_mult_E)
                        self.meta_cost += np.maximum(energy_rate[0] - self.ignore_energy, 0.0)  # energy_rate is a size-1 1D array
                        cr_in_r[10:15] -= residue_r
                        cr_in_r[9] = np.mean(self.ankle_vel_cache_long, axis=0)[1]      # TODO
                        energy_rate = neural_net_regress_elu(cr_in_r, self.EWmats, self.EBmats, input_mult=self.input_mult,
                                                             output_mult=self.output_mult_E)
                        # print(cr_in_r)
                        # print(energy_rate[0])
                        # if energy_rate[0] > 10000:
                        #     input("press enter to continue...")

                        self.meta_cost += np.maximum(energy_rate[0] - self.ignore_energy, 0.0)
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

            self.robot_skeleton.set_forces(tau_revised)

            # self.ankle_taus.append(tau_revised[10])
            # self.ankle_dq.append(self.robot_skeleton.dq[10])
            # self.ankle_q.append(self.robot_skeleton.q[10])
            # self.dy.append(self.robot_skeleton.dq[1])

            self.dart_world.step()
            s = self.state_vector()
            if not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all()):
                break

        self.total_residue = self.total_residue / 3 / 160.0 * self.residue_pen
        # print(self.total_residue)
        self.meta_cost = self.meta_cost / 6 * 10 / self.up_energy

    def advance(self, clamped_control):
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[6:] = clamped_control * self.action_scale

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

    def _step(self, a):
        # smoothly increase the target velocity
        self.target_vel = (np.min([self.t, self.tv_endtime]) / self.tv_endtime) * (
                self.final_tv - self.init_tv) + self.init_tv

        if self.t < self.assist_timeout:
            assert len(self.assist_schedule) > 0
            for sch in self.assist_schedule:
                if self.t > sch[0]:
                    self.current_pd = sch[1][0]
                    self.vel_enforce_kp = sch[1][1]

        posbefore = self.robot_skeleton.bodynode(self.push_target).com()[0]

        clamped_control = self.clamp_act(a)

        self.advance(np.copy(clamped_control))

        posafter = self.robot_skeleton.bodynode(self.push_target).com()[0]
        height = self.robot_skeleton.bodynode('head').com()[1]
        # side_deviation = self.robot_skeleton.bodynode('head').com()[2]
        side_deviation = self.robot_skeleton.com()[2]
        angle = self.robot_skeleton.q[3]

        upward = np.array([0, 1, 0])
        upward_world = self.robot_skeleton.bodynode('head').to_world(
            np.array([0, 1, 0])) - self.robot_skeleton.bodynode('head').to_world(np.array([0, 0, 0]))
        upward_world /= np.linalg.norm(upward_world)
        ang_cos_uwd = np.dot(upward, upward_world)
        ang_cos_uwd = np.arccos(ang_cos_uwd)

        forward = np.array([1, 0, 0])
        forward_world = self.robot_skeleton.bodynode('head').to_world(
            np.array([1, 0, 0])) - self.robot_skeleton.bodynode('head').to_world(np.array([0, 0, 0]))
        forward_world /= np.linalg.norm(forward_world)
        ang_cos_fwd = np.dot(forward, forward_world)
        ang_cos_fwd = np.arccos(ang_cos_fwd)

        lateral = np.array([0, 0, 1])
        lateral_world = self.robot_skeleton.bodynode('head').to_world(
            np.array([0, 0, 1])) - self.robot_skeleton.bodynode('head').to_world(np.array([0, 0, 0]))
        lateral_world /= np.linalg.norm(lateral_world)
        ang_cos_ltl = np.dot(lateral, lateral_world)
        ang_cos_ltl = np.arccos(ang_cos_ltl)

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        self.contact_info = np.array([0, 0])
        l_foot_force = np.array([0.0, 0, 0])
        r_foot_force = np.array([0.0, 0, 0])
        for contact in contacts:
            if contact.skel_id1 + contact.skel_id2 == 1:
                # not self-colliding
                if contact.bodynode1 == self.robot_skeleton.bodynode('l-foot') or \
                        contact.bodynode2 == self.robot_skeleton.bodynode('l-foot'):
                    self.contact_info[0] = 1
                    total_force_mag += np.linalg.norm(contact.force)
                    l_foot_force += contact.force
                if contact.bodynode1 == self.robot_skeleton.bodynode('r-foot') or \
                        contact.bodynode2 == self.robot_skeleton.bodynode('r-foot'):
                    self.contact_info[1] = 1
                    total_force_mag += np.linalg.norm(contact.force)
                    r_foot_force += contact.force

        vel = (posafter - posbefore) / self.dt

        self.vel_cache.append(vel)
        self.target_vel_cache.append(self.target_vel)

        if len(self.vel_cache) > int(2.0 / self.dt):
            self.vel_cache.pop(0)
            self.target_vel_cache.pop(0)

        vel_rew = -self.vel_reward_weight * np.abs(np.mean(self.target_vel_cache) - np.mean(self.vel_cache))
        if self.t < self.tv_endtime:
            vel_rew *= 0.5
        # TODO: does this make sense at all?
        if self.cur_step >= 300:
            vel_rew = vel_rew * (1.0 + (self.cur_step - 150) / 150.0 / self.final_tv)

        alive_bonus = self.alive_bonus_rew  # np.max([1.5 + self.final_tv * 0.5 * self.vel_reward_weight, 4.0])

        a_non_leg = np.copy(a[12:])
        a_leg = np.copy(a[:12])
        action_pen = 0.0
        if self.muscle_add_tor_limit and self.muscle_add_energy_cost:
            a_leg[0:5] = 0.0
            a_leg[6:11] = 0.0
            action_pen += self.energy_weight * self.meta_cost
        action_pen += self.energy_weight * np.abs(a_leg).sum()
        action_pen += 0.35 * np.abs(a_non_leg).sum()       # TODO: fix energy pen for non-legs

        deviation_pen = self.side_devia_weight * abs(side_deviation)

        # contact_pen = 0.5 * np.square(np.clip(l_foot_force, -2000, 2000) / 1000.0).sum() + np.square(np.clip(
        #     r_foot_force, -2000, 2000) / 1000.0).sum()

        rot_pen = self.rot_pen_weight * (0.3 * (abs(ang_cos_uwd)) + 0.3 * (abs(ang_cos_fwd)) + 1.5 * (abs(ang_cos_ltl)))

        spine_pen = self.abd_pen_weight * np.sum(np.abs(self.robot_skeleton.q[[18, 19]])) + self.spl_pen_weight * np.abs(
            self.robot_skeleton.q[20]) + self.angle_pen_weight * np.abs(self.robot_skeleton.q[3])

        # spine_pen += 0.05 * np.sum(np.abs(self.robot_skeleton.q[[8, 14]]))

        # dq_pen = 0.0 * np.sum(np.square(self.robot_skeleton.dq[6:]))

        # torso_vel_pen = 0.15*np.abs(self.robot_skeleton.bodynode('thorax').com_spatial_velocity()[0:3]).sum()

        # TODO: rot pen is currently not invloved
        reward = vel_rew + alive_bonus - action_pen - deviation_pen - spine_pen - rot_pen \
            - 0.5 * np.sum(np.abs(self.robot_skeleton.q[[11, 17]]))     # TODO   #- dq_pen # - contact_pen #+ stride_reward # - torso_vel_pen
        if self.muscle_add_tor_limit:
            reward -= 2 * np.sqrt(self.total_residue + 1) - 2

        self.t += self.dt
        self.cur_step += 1

        self.stepwise_rewards.append(reward)

        self.previous_control = clamped_control

        ob = self._get_obs()

        s = self.state_vector()
        broke_sim = False
        if not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all()):
            broke_sim = True

        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height - self.init_height > -0.35) and (height - self.init_height < 1.0) and (
                            abs(ang_cos_uwd) < 1.2) and (abs(ang_cos_fwd) < 1.2)
                    and np.abs(angle) < 1.2 and
                    np.abs(self.robot_skeleton.q[5]) < 1.2 and np.abs(self.robot_skeleton.q[4]) < 1.2 and np.abs(
                    self.robot_skeleton.q[3]) < 1.2
                    and np.abs(side_deviation) < 0.9)

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

        return ob, reward, done, {'broke_sim': broke_sim, 'vel_rew': vel_rew, 'action_pen': action_pen,
                                  'deviation_pen': deviation_pen, 'done_return': done,
                                  'dyn_model_id': 0, 'state_index': 0, 'com': self.robot_skeleton.com(),
                                  'contact_force': l_foot_force + r_foot_force,
                                  'contact_forces': [l_foot_force, r_foot_force],
                                  'avg_vel': np.mean(self.vel_cache), 'meta_cost':self.meta_cost
                                  }

    def _get_obs(self):
        state = np.concatenate([
            self.robot_skeleton.q[1:],
            self.robot_skeleton.dq,
        ])

        if self.include_additional_info:
            state = np.concatenate([state, self.contact_info])
            state = np.concatenate([state, [self.target_vel]])

        return state

    def reset_model(self):
        # print('resetttt')
        self.dart_world.reset()

        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.05, high=.05, size=self.robot_skeleton.ndofs)

        if self.final_tv > 3.0:     # running
            qpos[6] += self.np_random.uniform(low=-0.2, high=0.2)
        else:
            qpos[6] += self.np_random.uniform(low=-0.1, high=0.1)

        if self.add_flex_noise_q:
            qpos[19] += self.np_random.uniform(low=-0.2, high=0.0)
            if qpos[6] < 0:
                qpos[6] -= 0.1
            else:
                qpos[6] += 0.1

        qpos[10] = -qpos[6]
        qpos[12] = -qpos[6]
        qpos[16] = qpos[6]

        self.set_state(qpos, qvel)

        self.t = self.dt
        self.cur_step = 0
        self.stepwise_rewards = []

        self.contact_info = np.array([0, 0])

        self.previous_control = None

        self.vel_cache = []
        self.target_vel_cache = []

        self.dq_cache_l = []
        self.dq_cache_r = []
        self.ankle_vel_cache_long = []

        self.total_residue = 0
        assert len(self.assist_schedule) > 0
        self.current_pd = self.assist_schedule[0][1][0]
        self.vel_enforce_kp = self.assist_schedule[0][1][1]
        self.target_vel = 0.0
        self.meta_cost = 0.0

        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            # self.track_skeleton_id = 0
            self._get_viewer().scene.tb.trans[2] = -5.5