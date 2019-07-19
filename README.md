# lrle-rl-examples

Code for "Synthesis of Biologically Realistic Human Motion Using Joint Torque Actuation", SIGGRAPH 2019, Part 2

https://arxiv.org/abs/1904.13041

This repository contains examples applying the paper's techniques to deep RL locomotion training. For the techniques themselves (how to learn R and E), and for examples of using them in Optimal Control, go to the sibling repository: https://github.com/jyf588/lrle

## Installation: (reference: https://github.com/DartEnv/dart-env/wiki)

### 0. Clone this repo:
```bash
    git clone https://github.com/jyf588/lrle-rl-examples.git
    cd lrle-rl-examples
    git checkout master(or release-old, see Usage)
```

### 1. Install <a href="http://dartsim.github.io/">Dart</a>

#### Prerequisites For Mac OSX

```bash
    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    brew cask install xquartz
    brew install dartsim --only-dependencies
    brew install cmake
    brew install ode # ignore this if it tells you ode has already been installed
```

#### Prerequisites for Ubuntu

```bash
    sudo apt-get install build-essential cmake pkg-config git
    sudo apt-get install libeigen3-dev libassimp-dev libccd-dev libfcl-dev libboost-regex-dev libboost-system-dev
    sudo apt-get install libopenscenegraph-dev
    sudo apt-get install libbullet-dev
    sudo apt-get install liburdfdom-dev
    sudo apt-get install libnlopt-dev
    sudo apt-get install libxi-dev libxmu-dev freeglut3-dev
    sudo apt-get install libode-dev # ignore this if it tells you ode has already been installed
    sudo apt-get install libtinyxml2-dev
    sudo apt-get install libblacs-mpi-dev openmpi-bin
```

Note for Ubuntu 14.04 Trusty:

To correctly handle collision for capsule shape, ODE(Open Dynamics Engine) is required. However, currently, libode-dev seems to be broken on Trusty if installed from apt-get. To use capsule shape, please go to <a href="https://sourceforge.net/projects/opende/files//">ODE download</a> for installing ODE from source. 

#### Download and install Dart

```bash
    git clone git://github.com/dartsim/dart.git
    cd dart
    git checkout tags/v6.3.0
    cp ../patches/lcp.cpp dart/external/odelcpsolver/lcp.cpp
    mkdir build
    cd build
    cmake ..
    make -j4
    sudo make install
    cd ..
    cd ..
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

Note: If you have Nvidia drivers installed on your computer, you might need to do something similar to this https://github.com/RobotLocomotion/drake/issues/2087 to address one error during Dart installation.

### 2. Set up python environment

Install Anaconda first: https://www.anaconda.com, then:

```bash
    conda create -n lrle-rl-env python=3.6
    conda activate lrle-rl-env
```

Note: if you encouter permission denied errors below, try using "sudo chown" command to changed the privilege of the denied file/folder. It is in general bad practice to "sudo install/setup" something into a conda env.

### 3. Install <a href="http://pydart2.readthedocs.io/en/latest/">PyDart2</a>, a python binder for Dart

```bash
    conda install swig
    conda install pyqt=5
    git clone https://github.com/sehoonha/pydart2.git
    cd pydart2
    cp ../patches/pydart2_draw.cpp pydart2/pydart2_draw.cpp
```

Modify setup.py: add a space before -framework Cocoa; add add CXX_FLAGS += '-stdlib=libc++ ' after the line, CXX_FLAGS += '-framework GLUT ' (this is a temporary issue and should not be needed soon)

If you are using Xcode 10 on MacOS X Mojave, run the following line:

```
    MACOSX_DEPLOYMENT_TARGET=10.9 python setup.py build build_ext
```

otherwise:

```
    python setup.py build build_ext
```

And then:

```bash
    python setup.py develop
    export PYTHONPATH=$PWD:$PYTHONPATH
    cd ..
```

### 4. Install Dart Env, Openai Gym with Dart support

The installation is similar to <a href="https://github.com/openai/gym">**openai gym**</a>. To install, simply do 

```bash
    cd dart-env
    pip install -e .[dart]
    cd ..
```

Use **pip3** instead of **pip** if you plan to use python3.

### 5. Install Baselines, an deep reinforcement learning library
```bash
    cd baselines
    pip install -e .
    cd ..
```

### 6. Install Keras, to load trained neural net (R and E) easily
```bash
    pip install keras
```
Note: I would recommend install keras using pip rather than conda, since its dependency tensorflow was installed using pip as well when Baselines was installed.

### 7. Install OpenSim (Used only in the AMTU baseline)
```bash
    conda install -c kidzik opensim 
    conda install matplotlib
```

## Usage: (reference: https://github.com/VincentYu68/SymmetryCurriculumLocomotion)

### 1. Master branch:

There are two versions of the code in two separate branches. The master branch improves upon our results reported in the siggraph paper in terms of motion quality, by adding toes to the simulated human and better reward shaping. We no longer need curriculum training in this version.

Walking training: 
```bash
    cd baselines
    mpirun -np 8 python -m baselines.ppo1.run_humanoid_wtoe_walking --seed=some_number
```
Running training:
```bash
    mpirun -np 8 python -m baselines.ppo1.run_humanoid_wtoe_running --seed=some_number
```
Testing and visualizing trained policies:
```bash
    python test_policy.py DartHumanWalker-v2 PATH_TO_POLICY
```
The agent learns to walk/run in several hundred iterations, but letting it train longer usually gives more natural gaits.

Note: reading the following few files (instead of the whole repo) will suffice if you want to learn about the implementation details: baselines/baselines/ppo1/, dart-env/gym/envs/dart

### 2. Release-old branch:

This branch is thus deprecated and left here only for reproducing the siggraph paper results. The training here uses curriculum and follows exactly the same pipeline as in previous work: https://arxiv.org/abs/1801.08093

Walking training:
LR+LE:  
```
mpirun -np 8 python -m baselines.ppo1.run_humanoid_staged_learning --seed=xxx --HW_energy_weight=0.5 --HW_muscle_add_tor_limit=True --HW_muscle_add_energy_cost=True
```
LR: 
```
mpirun -np 8 python -m baselines.ppo1.run_humanoid_staged_learning --seed=xxx --HW_energy_weight=0.5 --HW_muscle_add_tor_limit=True 
```
BOX: 
```
mpirun -np 8 python -m baselines.ppo1.run_humanoid_staged_learning --seed=xxx --HW_energy_weight=0.5
```
AMTU: 
```
mpirun -np 8 python -m baselines.ppo1.run_humanoid_MD_staged_learning --seed=xxx --HW_energy_weight=0.5
```

Note 1: there is a bug with the argparse library: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse, and setting --HW_muscle_add_tor_limit=False will not work.

Note 2: argument —HW_energy_weight<0.4 will usually result in hopping motion

Running training:
LR+LE: 
```
mpirun -np 8 python -m baselines.ppo1.run_humanoid_staged_learning --seed=xxx --HW_energy_weight=0.45 --HW_muscle_add_tor_limit=True --HW_muscle_add_energy_cost=True --HW_final_tv=3.5 --HW_tv_endtime=2.0
```
LR: 
```
mpirun -np 8 python -m baselines.ppo1.run_humanoid_staged_learning --seed=xxx --HW_energy_weight=0.45 --HW_muscle_add_tor_limit=True  --HW_final_tv=3.5 --HW_tv_endtime=2.0
```
BOX: 
```
mpirun -np 8 python -m baselines.ppo1.run_humanoid_staged_learning --seed=xxx --HW_energy_weight=0.45 --HW_final_tv=3.5 --HW_tv_endtime=2.0
```
AMTU: fails to learn running

Testing and visualizing trained policy:
First change env.env.final_tv & env.env.tv_endtime in test_policy.py to match the values during training, then:
```bash
    python test_policy.py DartHumanWalker-v1 PATH_TO_POLICY
```
Or (for AMTU):
```bash
    python test_policy_MD.py DartHumanWalkerMD-v1 PATH_TO_POLICY
```
