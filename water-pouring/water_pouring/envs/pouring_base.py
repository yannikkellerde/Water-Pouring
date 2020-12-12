import gym
from gym import spaces
import pysplishsplash
import os,sys
import numpy as np
import random
import math
from collections import deque
from scipy.spatial.transform import Rotation as R
from time import perf_counter

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(FILE_PATH)
import utils.util as util
from utils.partio_utils import remove_particles
from utils.model3d import Model3d

class Pouring_base(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,use_gui=False,scene_file=os.path.join(FILE_PATH,"scenes","base_scene.json")):
        self.use_gui = use_gui
        self.scene_file = scene_file

        self.action_space = spaces.Box(low=0,high=1,shape=(4,))
        self.observation_space = spaces.Tuple((spaces.Box(low=-math.pi,high=math.pi,shape=(1,)),
                                              spaces.Box(low=-np.inf,high=np.inf,shape=(1,)),
                                              spaces.Box(low=-np.inf,high=np.inf,shape=(1,)),
                                              spaces.Box(low=0,high=1,shape=(1,)),
                                              spaces.Box(low=0,high=1,shape=(1,)),
                                              spaces.Box(low=0,high=np.inf,shape=(1,))))

        # Hyperparameters
        self.steps_per_action = 10
        self.rotation_uncertainty = 0
        self.translation_uncertainty = 0
        self.observation_uncertainty = 0
        self.max_rotation_radians = 0.002
        self.max_translation_x = 0.003
        self.max_translation_y = 0.003
        self.max_combined_power = 0.006
        self.base_translation_vector = np.array([self.max_translation_x, self.max_translation_y,0])
        self.human_saccade_time = 0.1
        self.max_observation_store = 5
        self.time_step_size = 0.005
        self.max_in_glas = 250
        self.fix_step = self.time_step_size/self.human_saccade_time
        self.reset()

    def seed(self,seed):
        np.random.seed(seed)

    def reset(self):
        remove_particles(os.path.join(FILE_PATH,"models/bigger/fluid.bgeo"),os.path.join(FILE_PATH,"models/bigger/tmp_fluid.bgeo"),0.5+random.random()/2)
        self.base = pysplishsplash.Exec.SimulatorBase()
        self.base.init(useGui=self.use_gui,outputDir=os.path.join(FILE_PATH,"particles"),sceneFile=self.scene_file)

        self.gui = None
        if self.use_gui:
            if self.gui is None:
                self.gui = pysplishsplash.GUI.Simulator_GUI_imgui(self.base)
            else:
                self.gui.die()
            self.base.setGui(self.gui)
        self.sim = pysplishsplash.Simulation()
        self.time_manager = pysplishsplash.TimeManager()
        self.time_manager.setTimeStepSize(self.time_step_size)
        self.base.initSimulation()
        self.base.initBoundaryData()

        self.bottle = Model3d(self.sim.getCurrent().getBoundaryModel(1).getRigidBodyObject())
        self.glas = Model3d(self.sim.getCurrent().getBoundaryModel(0).getRigidBodyObject())
        self.fluid = self.sim.getCurrent().getFluidModel(0)
        print("particle count:",self.fluid.numActiveParticles())

        self.particle_locations = self._get_particle_locations()

        self.fill_observations = deque(maxlen=self.max_observation_store)
        self.water_flow_observations = deque(maxlen=self.max_observation_store)
        self.fixation = 0
        self.fixation_direction = 0
        self.time = 0

    def _get_reward(self):
        def score_locations(locations):
            return locations["glas"]-locations["spilled"]*10
        new_locations = self._get_particle_locations()
        score = score_locations(new_locations)-score_locations(self.particle_locations)
        self.particle_locations = new_locations
        return score

    def _get_particle_locations(self):
        res = {
            "bottle":0,
            "glas":0,
            "air":0,
            "spilled":0
        }
        fluid_positions = []
        for i in range(self.fluid.numActiveParticles()):
            fluid_positions.append(self.fluid.getPosition(i))
        fluid_positions = np.array(fluid_positions)
        in_glas = self.glas.hull.find_simplex(fluid_positions)>=0
        res["glas"] = np.sum(in_glas)
        fluid_positions = fluid_positions[~in_glas]
        in_bottle = util.in_hull(fluid_positions,self.bottle.hull,self.bottle.orig_translation,
                                 self.bottle.new_translation,self.bottle.new_rotation)
        res["bottle"] = np.sum(in_bottle)
        fluid_positions = fluid_positions[~in_bottle]
        spilled = fluid_positions[:,1]<self.glas.orig_rect[1][1]
        res["spilled"] = np.sum(spilled)
        fluid_positions = fluid_positions[~spilled]
        res["air"] = len(fluid_positions)
        return res

    def step(self,action):
        """Move the bottle, to pour water into the glas.

        Parameters
        ----------
        action (np.array): A 4-dimensional vector of floats between 0 and 1 representing:
                    0: Rotation: For values <0.5 rotate the botte downwards, for >0.5 upwards.
                    1: Translation-x: For <0.5 move towards glas, for >0.5 away from glas.
                    2: Translation-y: For <0.5 move down, for >0.5, move up.
                    3: Fixation: For 0, observe fluid level in glas. For 1, observe rate of water flow.

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (6-tuple) :
                Entries represent in order: bottle-rotation, bottle-translation-x, bottle-translation-y,
                fixation/saccade location, estimated fluid level in glas, estimated rate of water flow.
            reward (float) :
                +1 for landing a water particle in the glas. -10 for spilling a water particle.
            episode_over (bool) :
                Whether it's time to reset the environment again.
            info (dict) :
                Some particle statistics
        """
        action = np.array(action)
        if len(action)!=4 or not np.all(action<=1) or not np.all(action>=0):
            raise ValueError(f"Invalid action {action}")
        rot_radians = (-action[0]+0.5)*2*self.max_rotation_radians
        to_translate = self.base_translation_vector*((np.array([action[1],action[2],0])-0.5)*2)
        mysum = abs(rot_radians)+np.sum(np.abs(to_translate))
        if mysum!=0:
            power_rate = self.max_combined_power/mysum
        else:
            power_rate = np.inf
        if power_rate<1:
            rot_radians*=power_rate
            to_translate*=power_rate
        to_rotate = R.from_euler("z",rot_radians).as_matrix()
        for step in range(self.steps_per_action):
            self.bottle.rotate(to_rotate)
            self.bottle.translate(to_translate)
            self.time += self.time_step_size
            self.base.timeStepNoGUI()
        self.bottle.body.updateVertices()
        old_particle_locations = self.particle_locations
        reward = self._get_reward()
        if self.fixation in (0,1):
            self.fixation_direction = -1 if action[3] < 0.5 else 1
        old_fix = self.fixation
        self.fixation += self.fixation_direction*self.time_step_size
        self.fixation = 0 if self.fixation<0 else (1 if self.fixation>1 else self.fixation)
        if self.fixation == 0:
            if old_fix!=0:
                self.fill_observations.clear()
            self.fill_observations.append(self.particle_locations["glas"])
        elif self.fixation == 1:
            if old_fix!=1:
                self.water_flow_observations.clear()
            self.water_flow_observations.append(old_particle_locations["bottle"]-self.particle_locations["bottle"])
        observation = self._observe()
        return observation,reward,self.time>10 or self.particle_locations["spilled"]>=10,self.particle_locations

    def render(self, mode='human'):
        if self.gui is None:
            raise Exception("Trying to render without initializing with gui_mode=True")
        self.gui.one_render()

    def _observe(self):
        rotation = R.from_matrix(self.bottle.rotation).as_euler("zyx")[0]
        translation_x,translation_y = self.bottle.translation[:2]
        if self.rotation_uncertainty != 0:
            rotation = np.random.normal(rotation,self.rotation_uncertainty)
        if self.translation_uncertainty != 0:
            translation_x = np.random.normal(translation_x,self.translation_uncertainty)
            translation_y = np.random.normal(translation_y,self.translation_uncertainty)
        water_flow_estimate = np.mean(self.water_flow_observations) if len(self.water_flow_observations)>0 else 0
        fill_state = np.mean(self.fill_observations)/self.max_in_glas if len(self.fill_observations)>0 else 0
        if self.observation_uncertainty != 0:
            water_flow_estimate = np.random.normal(water_flow_estimate,self.observation_uncertainty)
            fill_state = np.random.normal(fill_state,self.observation_uncertainty)
        return rotation,translation_x,translation_y,self.fixation,fill_state,water_flow_estimate

if __name__ == "__main__":
    env = Pouring_env(gui_mode=True)
    for _ in range(1000):
        obs,reward,done,info = env._step((0,0.5,0.5,0))
        env._render()
        print(obs,reward,done,info)

