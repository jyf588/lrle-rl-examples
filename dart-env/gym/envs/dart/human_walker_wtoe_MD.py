__author__ = 'yifengjiang'

import numpy as np
from gym import utils
from gym.envs.dart import dart_env
import os

import pydart2 as pydart

import opensim

from keras.models import load_model

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


def HillModel_RigidTendon(a, lMT, vMT, mus_params, j):
    ActiveFVParameters = np.array([-0.215812499592629,	-32.5966241739010,	-1.12412150864786,	0.912644059915004])
    PassiveFLParameters = np.array([-0.995172050006169,  53.5981500331442])
    Faparam = np.array([0.814483478343008, 1.05503342897057, 0.162384573599574, 0.0633034484654646,
                        0.433004984392647, 0.716775413397760, -0.0299471169706956, 0.200356847296188])

    # Fmax = mus_params['Fmax'][j]
    lMopt = mus_params['lMopt'][j]
    lTs = mus_params['lTs'][j]
    alphaopt = mus_params['alphaopt'][j]
    vMmax  = mus_params['vMax'][j]

    # multiply vMax(10.0) with lMopt
    vMmax = vMmax * lMopt

    # Hill-type muscle model: geometric relationships
    w = lMopt * np.sin(alphaopt)
    lM = np.sqrt((lMT-lTs)**2 + w**2) # Rigid Tendon: lT = lTs
    lMtilde = lM / lMopt
    lT = lTs * 1.0
    cos_alpha = (lMT-lT) / lM

    b11 = Faparam[0]
    b21 = Faparam[1]
    b31 = Faparam[2]
    b41 = Faparam[3]
    b12 = Faparam[4]
    b22 = Faparam[5]
    b32 = Faparam[6]
    b42 = Faparam[7]

    b13 = 0.1
    b23 = 1.0
    b33 = 0.5 * np.sqrt(0.5)
    b43 = 0.0
    num3 = lMtilde - b23
    den3 = b33 + b43*lMtilde
    FMtilde3 = b13*np.exp(-0.5*(num3**2) / (den3**2))

    num1 = lMtilde - b21
    den1 = b31 + b41*lMtilde
    FMtilde1 = b11*np.exp(-0.5*(num1**2) / (den1**2))

    num2 = lMtilde - b22
    den2 = b32 + b42*lMtilde
    FMtilde2 = b12*np.exp(-0.5*(num2**2) / (den2**2))

    FMactFL = FMtilde1 + FMtilde2 + FMtilde3

    # # TODO
    if cos_alpha < 0:
        cos_alpha = 0.0
    # Active muscle force-velocity relationship
    vMtilde = (vMT/vMmax) * cos_alpha
    e1 = 1.475 * ActiveFVParameters[0]
    e2 = 0.25 * ActiveFVParameters[1]
    e3 = ActiveFVParameters[2] + 0.75
    e4 = ActiveFVParameters[3] - 0.027

    FMactFV = e1 * np.log((e2*vMtilde+e3) + np.sqrt((e2*vMtilde+e3)**2+1)) + e4

    # Passive muscle force-length characteristic
    e0 = 0.6
    kpe = 4
    t5 = np.exp(kpe * (lMtilde - 1.0) / e0)
    FMpas = (t5  - PassiveFLParameters[0]) / PassiveFLParameters[1]

    return lMtilde, FMactFL, FMactFV, FMpas, cos_alpha


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


class DartHumanWalkerMDToeEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):

        # action/state related params

        # new action: (43+43+11): [:43] [43:86] left and right activations; [86:97] non-leg torques
        # control bounds are still set to [-1,1] for all 97 channels; and no change to clamp control
        # then for the first 86 clapmed controls x\in[-1,1], (x+1)/2 to make them [0,1] activations
        # query AMTU's MD_NN in do_simulation (every five steps)
        self.numMuscles = 43*2
        self.numTorActuator = 11
        act_dim = self.numMuscles + self.numTorActuator
        self.control_bounds = np.array([[1.0] * act_dim, [-1.0] * act_dim])

        self.action_scale_tor = np.array([150, 150, 100,
                                          100, 30, 15, 30, 100, 30, 15, 30])

        # leg_act_scale
        self.action_scale = np.array([200.0, 80, 80, 250, 250, 200, 80, 80, 250, 250])

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

        # MD init
        self.modelMD_path = 'neuralnets/MD_3D_-0915&-0606&-0606&-2101&-0909_dq555&10&15_tau250&80&80&250&250_2392_jul14_25618080elu_mse.h5'
        self.MDWmats = []
        self.MDBmats = []
        self.MDWmats, self.MDBmats = load_model_weights(self.modelMD_path)

        self.input_mult = np.ones([int(self.numMuscles / 2 + 10)])
        self.input_mult[-5:-1] /= 2.5   # dqs
        self.input_mult[-1:] /= 7.5
        self.dq_clip = np.array([5.0, 5, 5, 10, 15])
        self.dq_cache_l = []
        self.dq_cache_r = []

        # # TODO: for DEBUG only
        # fullpath = os.path.join(os.path.dirname(__file__), "assets", self.modelMD_path)
        # self.modelMD = load_model(fullpath)
        # fullpath = os.path.join(os.path.dirname(__file__), "assets", self.modelE_path)
        # self.modelE = load_model(fullpath)
        # self.ankle_taus = []
        # self.ankle_dq = []
        # self.ankle_q = []
        # self.dy = []
        # self.residues = []

        # reward related
        self.energy_weight = 0.3  # set outside
        self.alive_bonus_rew = 9.0  # set outside
        self.vel_reward_weight = 7.0  # set outside
        self.side_devia_weight = 1.5  # set outside
        self.jl_pen_weight = 0.7  # set outside
        self.alive_pen = 0.0  # set outside
        self.use_muscle_based_cost = True  # TODO
        self.up_energy = 800.0

        self.final_tar_v = 1.4  # set outside
        self.tar_acc_time = 1.7  # set outside
        self.target_vel = None
        self.vel = None
        self.vel_cache = []

        self.n_tau_leg_10 = None

        # init opensim muscle model
        self.mus_params = None
        self.model_os = None
        self.model_state = None
        self.coord_set = None
        self.coord_index = None
        self.mus_index = None
        self.dm_mask = None
        self.init_osim()

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

    def init_osim(self):
        pathname = 'others/Mus_Coord_OI.mat'
        from os import path
        if pathname.startswith("/"):
            fullpath = pathname
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", pathname)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        import scipy.io as sio
        mat_contents = sio.loadmat(fullpath)

        MusclesOI = mat_contents['MusclesOI']
        Fmax_arr = [j[0, 0] for j in MusclesOI['maxIsoForce'].reshape(-1)]  # odd workaround
        lMopt_arr = [j[0, 0] for j in MusclesOI['optFL'].reshape(-1)]
        lTs_arr = [j[0, 0] for j in MusclesOI['tendonSL'].reshape(-1)]
        alphaopt_arr = [j[0, 0] for j in MusclesOI['alpha'].reshape(-1)]
        vMax_arr = [j[0, 0] for j in MusclesOI['maxConVel'].reshape(-1)]

        self.mus_params = {'Fmax': Fmax_arr, 'lMopt': lMopt_arr, 'lTs': lTs_arr, 'alphaopt': alphaopt_arr, 'vMax': vMax_arr}

        pathname = 'others/gait2392_simbody.osim'
        if pathname.startswith("/"):
            fullpath = pathname
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", pathname)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.model_os = opensim.Model(fullpath)
        self.model_state = self.model_os.initSystem()
        self.coord_set = self.model_os.getCoordinateSet()

        self.coord_index = np.arange(13, 18, 1, dtype=int)
        self.mus_index = np.arange(43, 86, 1, dtype=int)

        # # kinematics = np.zeros([10], dtype=float)
        #
        # kinematics = np.array([-1.17667983508164,
        #                        0.192864303902365,
        #                        -0.446409470007429,
        #                        -0.843609197093483,
        #                        1.20276376677630,
        #                        -0.0879652747409948,
        #                        0.989407563655357,
        #                        -0.337088300777318,
        #                        -0.505929855046541,
        #                        -1.44678903208270])

        # nMus = mus_index.shape[0]
        # nDof = coord_index.shape[0]
        # a = np.ones((nMus, 1)) / 2.0

        self.dm_mask = np.array([[1, 1, 1, 0, 0],
                                [1, 1, 1, 0, 0],
                                [1, 1, 1, 0, 0],
                                [1, 1, 1, 0, 0],
                                [1, 1, 1, 0, 0],
                                [1, 1, 1, 0, 0],
                                [1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 0],
                                [0, 0, 0, 1, 0],
                                [1, 1, 1, 1, 0],
                                [1, 1, 1, 0, 0],
                                [1, 1, 1, 0, 0],
                                [1, 1, 1, 0, 0],
                                [1, 1, 0, 0, 0],  # 15,3 is problematic
                                [1, 1, 1, 0, 0],
                                [1, 1, 1, 1, 0],
                                [1, 1, 1, 0, 0],
                                [1, 1, 1, 1, 0],
                                [1, 1, 1, 0, 0],
                                [1, 1, 1, 0, 0],
                                [1, 1, 1, 0, 0],
                                [1, 1, 1, 0, 0],
                                [1, 1, 1, 0, 0],
                                [1, 1, 1, 0, 0],
                                [1, 1, 1, 0, 0],
                                [1, 1, 1, 0, 0],
                                [1, 1, 1, 1, 0],
                                [0, 0, 0, 1, 0],
                                [0, 0, 0, 1, 0],
                                [0, 0, 0, 1, 0],
                                [0, 0, 0, 1, 1],
                                [0, 0, 0, 1, 1],
                                [0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 1]])

    def init_params(self, args):
        if args is not None:
            for arg in vars(args):
                if arg[:3] == 'HW_' and hasattr(self, arg[3:]):
                    setattr(self, arg[3:], getattr(args, arg))

        # print(self.__dict__)

    def compute_muscle_torque(self, acts_l, q_l, dq_l):
        nMus = self.mus_index.shape[0]
        nDof = self.coord_index.shape[0]

        LMT = np.zeros((nMus, 1))
        VMT = np.zeros((nMus, 1))
        dM = np.zeros((nMus, nDof))

        for q_ind in range(0, nDof):
            c_ind = int(self.coord_index[q_ind])
            state_value = q_l[q_ind]
            self.coord_set.get(c_ind).setValue(self.model_state, state_value)
            state_value = dq_l[q_ind]
            self.coord_set.get(c_ind).setSpeedValue(self.model_state, state_value)

        # We need to realize the multibody system (which is part of SimBody and
        # is thus not part of the API exposed to Matlab). We need an opensim
        # function that realizes velocity without possibly violating any of the
        # constraints in the model, as this would induce an error.
        self.model_os.computeStateVariableDerivatives(self.model_state)

        # Model_OS.equilibrateMuscles(state); # Can't use this as some kinematic states might induce errors in the OpenSim source code.

        # Now that the model is in the correct state we can read out the muscle
        # information
        for i in range(0, nMus):
            m_ind = int(self.mus_index[i])
            muscle = self.model_os.getMuscles().get(m_ind)
            LMT[i, 0] = muscle.getLength(self.model_state)
            VMT[i, 0] = muscle.getLengtheningSpeed(self.model_state)
            for k in range(0, nDof):
                if self.dm_mask[i, k] == 1:
                    c_ind = int(self.coord_index[k])
                    dM[i, k] = muscle.computeMomentArm(self.model_state, self.coord_set.get(c_ind))

        FMltilde = np.ones((nMus, 1))  # Normalized Force-Length dependence
        FMvtilde = np.ones((nMus, 1))  # Normalized Force-Velocity dependence
        Fpe = np.ones((nMus, 1))  # Passive (elastic) force
        cos_alpha = np.ones((nMus, 1))  # cosine of the muscles pennation angle

        # We apply the Hill Model under the assumption of a rigid tendon and
        # calculate the normalized quantities of passive and active muscle force
        # components. The active component will be scaled with the muscle
        # activation and added to the passive force to produce the final muscle force.
        for m in range(0, nMus):
            _, FMltilde[m, 0], FMvtilde[m, 0], Fpe[m, 0], cos_alpha[m, 0] =\
                HillModel_RigidTendon(1.0, LMT[m, 0], VMT[m, 0], self.mus_params, m)

        FMo = np.array(self.mus_params['Fmax'])[:, np.newaxis]

        Fpas = FMo * Fpe * cos_alpha
        Fact = FMo * FMltilde * FMvtilde * cos_alpha
        FM = acts_l.reshape(-1,1) * Fact + Fpas
        tau = np.dot(dM.T, FM)

        return tau.reshape(-1)

    def do_simulation(self, c_control, n_frames):
        acts = (c_control[:self.numMuscles] + 1.0) / 2
        acts_nn = acts - 0.5    # what input to the NN will need to be centered to -0.5 ~ 0.5
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau_non_muscle = c_control[self.numMuscles:self.numMuscles+self.numTorActuator] * self.action_scale_tor
        tau[(6+12):] = tau_non_muscle

        self.n_tau_leg_10 = 0

        for frame in range(n_frames):
            # print('frame',frame)
            self.dq_cache_l.append(self.robot_skeleton.dq[6:11])
            self.dq_cache_r.append(self.robot_skeleton.dq[12:17])
            if len(self.dq_cache_l) > 5:
                self.dq_cache_l.pop(0)
                self.dq_cache_r.pop(0)

            if frame % 5 == 0:
                # infer 10 leg torque from activations
                dq_in_l = np.clip(np.mean(self.dq_cache_l, axis=0), -self.dq_clip, self.dq_clip)
                md_in_l = np.concatenate([acts_nn[:int(self.numMuscles/2)], self.robot_skeleton.q[6:11], dq_in_l])
                tau_muscle_l = neural_net_regress_elu(md_in_l, self.MDWmats, self.MDBmats, input_mult=self.input_mult)
                # print(tau_muscle_l)
                # print(self.compute_muscle_torque(acts[:int(self.numMuscles/2)], self.robot_skeleton.q[6:11], dq_in_l))
                # tau_muscle_l = self.compute_muscle_torque(acts[:int(self.numMuscles/2)], self.robot_skeleton.q[6:11], dq_in_l)
                tau[6:11] = np.clip(tau_muscle_l, -self.action_scale[0:5], self.action_scale[0:5])

                dq_in_r = np.clip(np.mean(self.dq_cache_r, axis=0), -self.dq_clip, self.dq_clip)
                md_in_r = np.concatenate([acts_nn[int(self.numMuscles/2):], self.robot_skeleton.q[12:17], dq_in_r])
                tau_muscle_r = neural_net_regress_elu(md_in_r, self.MDWmats, self.MDBmats, input_mult=self.input_mult)
                # print(tau_muscle_r)
                # print(self.compute_muscle_torque(acts[int(self.numMuscles/2):], self.robot_skeleton.q[12:17], dq_in_r))
                # tau_muscle_r = self.compute_muscle_torque(acts[int(self.numMuscles/2):], self.robot_skeleton.q[12:17], dq_in_r)
                tau[12:17] = np.clip(tau_muscle_r, -self.action_scale[5:10], self.action_scale[5:10])

                # make this similar to np.abs(a).sum()
                self.n_tau_leg_10 += np.abs(tau_muscle_l / self.action_scale[0:5]).sum()
                self.n_tau_leg_10 += np.abs(tau_muscle_r / self.action_scale[5:10]).sum()

                # print("RR:")
                # print(md_in_l)
                # print(tau[6:11])
                # md_in_l = md_in_l * self.input_mult
                # print(self.modelMD.predict(md_in_l[np.newaxis, :], batch_size=1))
                # print(md_in_r)
                # print(tau[12:17])
                # md_in_r = md_in_r * self.input_mult
                # print(self.modelMD.predict(md_in_r[np.newaxis, :], batch_size=1))

            self.robot_skeleton.set_forces(tau)

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

            self.dart_world.step()
            if self.is_broke_sim():
                break

        self.n_tau_leg_10 /= 3.0    # since frame skip is 15 (5*3)

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
        return not (np.isfinite(s).all() and (np.abs(s[3:]) < 20).all())

    def _step(self, a):
        # smoothly increase the target velocity
        posbefore = self.robot_skeleton.com()[0]

        clamped_control = self.clamp_act(a)
        self.do_simulation(np.copy(clamped_control), self.frame_skip)

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

        action_pen = 0.0
        if self.use_muscle_based_cost:
            FMo = np.array(self.mus_params['Fmax'])
            lMopt = np.array(self.mus_params['lMopt'])
            acts = (clamped_control[:self.numMuscles] + 1.0) / 2
            acts_l = acts[:int(self.numMuscles/2)]
            acts_r = acts[int(self.numMuscles/2):]
            meta_cost = np.sum(acts_l * FMo * lMopt + acts_r * FMo * lMopt)   # always positive
            # *10 to match scale of n_tau_leg_10, up_energy is normalize factor
            action_pen += self.energy_weight * meta_cost * 10 / self.up_energy
        else:
            action_pen += self.energy_weight * self.n_tau_leg_10
        action_pen += np.minimum(self.energy_weight, 0.4) * np.abs(a[self.numMuscles:]).sum()

        deviation_pen = self.side_devia_weight * abs(side_deviation)

        reward = - action_pen - deviation_pen - self.calc_head_pen() - self.calc_joint_limits_pen()
        reward += vel_reward
        reward += alive_bonus

        # directly regularize activation a bit
        reward -= 0.005 * np.sum(np.abs((a[:self.numMuscles] + 1.0) / 2))

        self.previous_control = clamped_control

        ob = self._get_obs()

        broke_sim = self.is_broke_sim()
        alive = (height - self.init_height > -0.3) and (height - self.init_height < 1.0) and (
                np.abs(side_deviation) < 0.9) and (not broke_sim)
        done = not alive

        # if self.cur_step == 499:
        #     plt.figure()
        #     plt.title('residues')
        #     plt.plot(self.residues, label='residues')
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
                                  'avg_vel': np.mean(self.vel_cache)
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

        self.n_tau_leg_10 = None

        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            # self.track_skeleton_id = 0
            self._get_viewer().scene.tb.trans[2] = -5.5
