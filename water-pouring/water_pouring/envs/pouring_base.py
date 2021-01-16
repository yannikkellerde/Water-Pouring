import gym
from gym import spaces
import pysplishsplash
import os,sys
import time
import numpy as np
import random
import math
from collections import deque
from scipy.spatial.transform import Rotation as R
from scipy.stats import norm
from time import perf_counter

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(FILE_PATH)
import pouring_utils.util as util
from pouring_utils.partio_utils import remove_particles, count_particles
from pouring_utils.model3d import Model3d

class Pouring_base(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,use_gui=False,obs_uncertainty=0,scene_base=os.path.join(FILE_PATH,"scenes","simple_scene.json"),glas="normal.obj"):
        self.use_gui = use_gui
        self.scene_file = os.path.join(FILE_PATH,"scenes","tmp_scene.json")
        util.manip_scene_file(scene_base,self.scene_file,env=self,glas=glas)

        # Gym
        self.action_space = spaces.Box(low=-1,high=1,shape=(3,))
        self.observation_space = spaces.Box(low=0,high=1,shape=(6,))

        # Hyperparameters
        ## Uncertainty
        self.obs_uncertainty = obs_uncertainty
        self.proposal_function_rate = 0.05

        ## Actions and Movement
        self.translation_bounds = ((-0.5,1),(0,1.5))
        self.max_translation_x = 0.0015
        self.max_translation_y = 0.0015
        self.base_translation_vector = np.array([self.max_translation_x, self.max_translation_y,0])
        self.max_rotation_radians = 0.002
        self.min_rotation = 1.22

        ## Rewards
        self.time_step_punish = 0.1
        self.spill_punish = 15
        self.max_spill = 15
        self.hit_reward = 1
        self.temperature = 1

        ## Features
        self.max_observation_store = 5

        ## Simulation
        self.steps_per_action = 20
        self.time_step_size = 0.005
        self._max_episode_steps = ((1/self.time_step_size)*30)/self.steps_per_action # 30 seconds
        self.max_in_glas = 250
        self.fluid_base_path = os.path.join(FILE_PATH,"models/fluids/fluid.bgeo")
        self.max_particles = count_particles(self.fluid_base_path)

        self.gui = None
        self.reset(first=True)

    def seed(self,seed):
        np.random.seed(seed)

    def _walk_uncertaintys(self):
        if self.obs_uncertainty == 0:
            return
        gaussian = norm(0,self.obs_uncertainty).pdf
        for key in self.current_walk:
            proposal_accepted = False
            while not proposal_accepted:
                proposal = np.random.normal(self.current_walk[key],self.obs_uncertainty*self.proposal_function_rate)
                cur_height = gaussian(self.current_walk[key])
                if random.random() < gaussian(proposal)/cur_height:
                    proposal_accepted = True
            self.current_walk[key] = proposal

    def reset(self,first=False,use_gui=None):
        if not first:
            print("In glas:",self.particle_locations["glas"])
        if not first:
            self.base.cleanup()
        if self.gui is not None:
            self.gui.die()
        self.gui = None
        if use_gui is not None:
            self.use_gui = use_gui
        self.base = pysplishsplash.Exec.SimulatorBase()
        self.base.init(useGui=self.use_gui,outputDir=os.path.join(FILE_PATH,"particles"),sceneFile=self.scene_file)
        if self.use_gui:
            if self.gui is None:
                self.gui = pysplishsplash.GUI.Simulator_GUI_imgui(self.base)
        if self.use_gui:
            self.base.setGui(self.gui)
        self.sim = pysplishsplash.Simulation()
        self.base.initSimulation()
        self.base.initBoundaryData()

        self.bottle = Model3d(self.sim.getCurrent().getBoundaryModel(1).getRigidBodyObject())
        self.glas = Model3d(self.sim.getCurrent().getBoundaryModel(0).getRigidBodyObject(),stretch_vertices=0.1)
        self.fluid = self.sim.getCurrent().getFluidModel(0)
        #print("particle count:",self.fluid.numActiveParticles())

        self.particle_locations = self._get_particle_locations()

        self.fill_observation = 0
        self.water_flow_observations = deque(maxlen=self.max_observation_store)
        if self.obs_uncertainty == 0:
            self.current_walk = {
                "rotation":0,
                "translation_x":0,
                "translation_y":0,
                "fill_estimate":0,
                "water_flow":0,
                "bottle_fill":0
            }
        else:
            self.current_walk = {
                "rotation":np.random.normal(0,self.obs_uncertainty),
                "translation_x":np.random.normal(0,self.obs_uncertainty),
                "translation_y":np.random.normal(0,self.obs_uncertainty),
                "fill_estimate":np.random.normal(0,self.obs_uncertainty),
                "water_flow":np.random.normal(0,self.obs_uncertainty),
                "bottle_fill":np.random.normal(0,self.obs_uncertainty)
            }
        self.time = 0
        self._step_number = 0
        self.done = False
        return self._observe()

    def _get_reward(self):
        def score_locations(locations):
            return np.array((locations["glas"]*self.hit_reward,min(locations["spilled"],self.max_spill)*self.spill_punish))
        new_locations = self._get_particle_locations()
        reward,punish = score_locations(new_locations)-score_locations(self.particle_locations)
        punish += self.time_step_punish
        true_score = reward-punish
        score = reward-punish*self.temperature
        self.particle_locations = new_locations
        return score,true_score

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
        in_bottle = util.in_hull(fluid_positions,self.bottle.hull,self.bottle.orig_translation,
                                 self.bottle.new_translation,self.bottle.new_rotation)
        res["bottle"] = np.sum(in_bottle)
        fluid_positions = fluid_positions[~in_bottle]
        in_glas = self.glas.hull.find_simplex(fluid_positions)>=0
        res["glas"] = np.sum(in_glas)
        fluid_positions = fluid_positions[~in_glas]
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

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (6-tuple) :
                Entries represent in order: bottle-rotation, bottle-translation-x, bottle-translation-y,
                bottle_fill, estimated fluid level in glas, estimated rate of water flow.
            reward (float) :
                +1 for landing a water particle in the glas. -10 for spilling a water particle.
            episode_over (bool) :
                Whether it's time to reset the environment again.
            info (dict) :
                Some particle statistics
        """
        action = np.array(action)
        punish = 0
        if len(action)!=3:
            raise ValueError("Invalid action {}".format(action))
        for i,a in enumerate(action):
            if a<-1:
                punish -= a
                action[i] = -1
            elif a>1:
                punish += a-1
                action[i] = 1
        self._step_number += 1
        rot_radians = -action[0]*self.max_rotation_radians
        to_translate = self.base_translation_vector*np.array([action[1],action[2],0],dtype=np.float)

        if (self.bottle.translation[0] + to_translate[0]*self.steps_per_action > self.translation_bounds[0][1] or
                self.bottle.translation[0] + to_translate[0]*self.steps_per_action < self.translation_bounds[0][0] or
                self.bottle.translation[1] + to_translate[1]*self.steps_per_action > self.translation_bounds[1][1] or
                self.bottle.translation[1] + to_translate[1]*self.steps_per_action < self.translation_bounds[1][0]):
            self.done = True
            punish += 200
        if (0<R.from_matrix(self.bottle.rotation).as_euler("zyx")[0]<self.min_rotation):
            self.done = True
            if (self.particle_locations["glas"]==0):
                punish += 200
        bottle_radians = R.from_matrix(self.bottle.rotation).as_euler("zyx")[0]
        if bottle_radians + rot_radians*self.steps_per_action > math.pi:
            rot_radians = 0
        tt,rr = self.glas.check_if_in_rect(self.bottle,to_translate*self.steps_per_action,rot_radians*self.steps_per_action)
        to_translate = tt/self.steps_per_action
        rot_radians = rr/self.steps_per_action
        to_rotate = R.from_euler("z",rot_radians).as_matrix()
        for step in range(self.steps_per_action):
            self.bottle.rotate(to_rotate)
            self.bottle.translate(to_translate)
            self.time += self.time_step_size
            self.base.timeStepNoGUI()
        self.bottle.body.updateVertices()
        old_particle_locations = self.particle_locations
        reward,true_reward = self._get_reward()
        reward-=punish
        true_reward-=punish
        self.fill_observation = self.particle_locations["glas"]
        self.water_flow_observations.append(old_particle_locations["bottle"]-self.particle_locations["bottle"])
        self._walk_uncertaintys()
        observation = self._observe()
        if (self._step_number>self._max_episode_steps or 
            self.particle_locations["spilled"]>=self.max_spill):
            self.done = True
        return observation,reward,self.done,{"true_reward":true_reward}

    def render(self, mode='human'):
        if self.gui is None:
            raise Exception("Trying to render without initializing with gui_mode=True")
        self.gui.one_render()

    def _normalize_observation(self,rotation,translation_x,translation_y,in_bottle,fill_state,water_flow_estimate):
        rotation = (rotation-self.min_rotation)/(math.pi-self.min_rotation)
        translation_x = (translation_x - self.translation_bounds[0][0]) / (self.translation_bounds[0][1]-self.translation_bounds[0][0])
        translation_y = (translation_y - self.translation_bounds[1][0]) / (self.translation_bounds[1][1]-self.translation_bounds[1][0])
        fill_state = fill_state/self.max_in_glas
        water_flow_estimate = water_flow_estimate/self.max_water_flow
        in_bottle = in_bottle/self.max_particles

        rotation = rotation + self.current_walk["rotation"]
        translation_x = translation_x + self.current_walk["translation_x"]
        translation_y = translation_y + self.current_walk["translation_y"]
        water_flow_estimate = water_flow_estimate + self.current_walk["water_flow"]
        fill_state = fill_state + self.current_walk["fill_estimate"]
        in_bottle = in_bottle + self.current_walk["bottle_fill"]

        rotation = np.clip(rotation,0,1)
        translation_x = np.clip(translation_x,0,1)
        translation_y = np.clip(translation_y,0,1)
        water_flow_estimate = np.clip(water_flow_estimate,0,1)
        fill_state = np.clip(fill_state,0,1)
        in_bottle = np.clip(in_bottle,0,1)
        return rotation,translation_x,translation_y,in_bottle,fill_state,water_flow_estimate


    def _observe(self):
        rotation = R.from_matrix(self.bottle.rotation).as_euler("zyx")[0]
        translation_x,translation_y = self.bottle.translation[:2]
        water_flow_estimate = np.mean(self.water_flow_observations) if len(self.water_flow_observations)>0 else 0
        fill_state = self.fill_observation
        in_bottle = self.particle_locations["bottle"]
        obs = self._normalize_observation(rotation,translation_x,translation_y,in_bottle,fill_state,water_flow_estimate)
        return np.array(obs)

if __name__ == "__main__":
    env = Pouring_env(gui_mode=True)
    for _ in range(1000):
        obs,reward,done,info = env._step((0,0.5,0.5,0))
        env._render()
        print(obs,reward,done,info)

