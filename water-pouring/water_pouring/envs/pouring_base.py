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
    def __init__(self,use_gui=False,fixed_spill=False,fixed_tsp=False,fixed_target_fill=False,obs_uncertainty=0,policy_uncertainty=0,jerk_punish=0,scene_base=os.path.join("scenes","simple_scene.json"),glas="normal.obj"):
        self.scene_base = os.path.join(FILE_PATH,scene_base)
        self.use_gui = use_gui
        self.fixed_tsp = fixed_tsp
        self.fixed_spill = fixed_spill
        self.fixed_target_fill = fixed_target_fill
        self.time_step_punish = 1
        self.spill_punish = 15
        self.scene_file = os.path.join(FILE_PATH,"scenes","tmp_scene.json")
        util.manip_scene_file(self.scene_base,self.scene_file,env=self,glas=glas)
        #remove_particles(os.path.join(FILE_PATH,"models","fluids","fluid.bgeo"),os.path.join(FILE_PATH,"models","fluids","tmp_fluid.bgeo"),0.55)

        # Gym
        self.action_space = None
        self.observation_space = None

        # Hyperparameters
        ## Uncertainty
        self.policy_uncertainty = policy_uncertainty
        self.obs_uncertainty = obs_uncertainty
        self.proposal_function_rate = 0.05

        ## Actions and Movement
        self.translation_bounds = ((-0.1,0.2),(-0.04,0.3))
        self.max_translation_x = 4e-4
        self.max_translation_y = 4e-4
        self.base_translation_vector = np.array([self.max_translation_x, self.max_translation_y,0])
        self.max_rotation_radians = 0.003
        self.min_rotation = 1.22

        ## Rewards
        self.time_step_punish_range = [0,2]
        self.spill_range = [1,50]
        self.max_spill = 20
        self.hit_reward = 1
        self.jerk_punish = jerk_punish


        ## Simulation
        self.steps_per_action = 20
        self.time_step_size = 0.005
        self._max_episode_steps = ((1/self.time_step_size)*20)/self.steps_per_action # 20 seconds
        self.max_in_glas = 390
        self.target_fill_range = [30,self.max_in_glas]
        self.target_fill_state = self.max_in_glas
        self.fluid_base_path = os.path.join(os.path.dirname(self.scene_file),util.get_fluid_path(self.scene_file))
        self.max_particles = count_particles(self.fluid_base_path)
        print(self.max_particles)

        self.gui = None
        self.reset(first=True)

    def seed(self,seed):
        np.random.seed(seed)

    def _get_random(self,attr_range):
        return random.random() * (attr_range[1]-attr_range[0]) + attr_range[0]

    def reset(self,first=False,use_gui=None):
        if not first:
            print("In glas:",self.particle_locations["glas"])
            print("Spilled:",self.particle_locations["spilled"])
            print("Spill punish",self.spill_punish)
            print("Max spill",self.max_spill)
            print("TSP",self.time_step_punish)
            print("Target fill",self.target_fill_state)
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

        self.particle_locations = self._get_particle_locations()
        if not self.fixed_tsp:
            self.time_step_punish = self._get_random(self.time_step_punish_range)
        if not self.fixed_spill:
            self.spill_punish = self._get_random(self.spill_range)
        if not self.fixed_target_fill:
            self.target_fill_state = self._get_random(self.target_fill_range)

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
        self.last_actions = deque([[0,0,0] for i in range(3)],maxlen=3)
        return self._observe()

    def _score_locations(self,locations,target_fill_state,spill_punish,max_spill):
        hit_reward = (self.max_in_glas/target_fill_state)*self.hit_reward
        return np.array((target_fill_state-abs(locations["glas"]-target_fill_state)*hit_reward,min(locations["spilled"],max_spill)*spill_punish))

    def _imagine_reward(self,time_step_punish,spill_punish,target_fill_state):
        new_locations = self._get_particle_locations()
        reward,punish = (self._score_locations(new_locations,target_fill_state,spill_punish,self.max_spill) -
                         self._score_locations(self.particle_locations,target_fill_state,spill_punish,self.max_spill))
        punish += time_step_punish
        if self.jerk_punish>0:
            pun_jerk = 0
            action_np = np.array(self.last_actions)
            for i in range(self.action_space.shape[0]):
                pun_jerk += self.jerk_punish*(util.approx_2rd_deriv(*action_np[:,i],self.time_step_size*self.steps_per_action)**2)
            if pun_jerk > 10:
                pun_jerk = 10
            punish += pun_jerk
        score = reward-punish
        self.particle_locations = new_locations
        return score

    def _get_reward(self):
        return self._imagine_reward(self.time_step_punish,self.spill_punish,self.target_fill_state)

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
        self.last_actions.appendleft(action)
        punish = 0
        if len(action)!=3 or np.any(action>1) or np.any(action<-1):
            raise ValueError("Invalid action {}".format(action))
        if self.policy_uncertainty>0:
            action = np.random.normal(action,np.abs(action)*self.policy_uncertainty)
        action = np.clip(action,-1,1)
        self._step_number += 1
        rot_radians = -action[0]*self.max_rotation_radians
        to_translate = self.base_translation_vector*np.array([action[1],action[2],0],dtype=np.float)

        if (self.bottle.translation[0] + to_translate[0]*self.steps_per_action > self.translation_bounds[0][1] or
                self.bottle.translation[0] + to_translate[0]*self.steps_per_action < self.translation_bounds[0][0]):
            to_translate[0] = 0
        
        if (self.bottle.translation[1] + to_translate[1]*self.steps_per_action > self.translation_bounds[1][1] or
                self.bottle.translation[1] + to_translate[1]*self.steps_per_action < self.translation_bounds[1][0]):
            to_translate[1] = 0
        if ((R.from_matrix(self.bottle.rotation).as_euler("zyx")[0]<self.min_rotation) and self.particle_locations["air"]==0 and self.particle_locations["glas"]!=0):
            self.done = True
            #if (self.particle_locations["glas"]==0):
            #    punish += 500
            # else:
            #     punish -= 50
        if (self._step_number>self._max_episode_steps):
            #punish += 100
            self.done = True
        if (self.particle_locations["spilled"]>=self.max_spill):
            self.done = True
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
        reward = self._get_reward()
        reward-=punish
        observation = self._observe()
        return observation,reward,self.done,{}

    def render(self, mode='human'):
        if self.gui is None:
            raise Exception("Trying to render without initializing with gui_mode=True")
        self.gui.one_render()

    def _observe(self):
        raise NotImplementedError("Observation not implemented")

if __name__ == "__main__":
    env = Pouring_env(gui_mode=True)
    for _ in range(1000):
        obs,reward,done,info = env._step((0,0.5,0.5,0))
        env._render()
        print(obs,reward,done,info)

