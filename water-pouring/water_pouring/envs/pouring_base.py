import gym
from gym import spaces
import pysplishsplash
import os,sys
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
    def __init__(self,use_gui=False,uncertainty=0,scene_file=os.path.join(FILE_PATH,"scenes","base_scene.json")):
        self.use_gui = use_gui
        self.scene_file = scene_file

        self.action_space = spaces.Box(low=0,high=1,shape=(4,))
        self.observation_space = spaces.Box(low=0,high=1,shape=(7,))

        # Hyperparameters
        self.steps_per_action = 10
        self.uncertainty = uncertainty
        self.fluid_base_path = os.path.join(FILE_PATH,"models/bigger/fluid.bgeo")
        self.translation_bounds = ((-2,2),(-2,2))
        self.proposal_function_rate = 0.05
        self.max_rotation_radians = 0.004
        self.max_translation_x = 0.006
        self.max_translation_y = 0.006
        self.max_combined_power = 0.01
        self.base_translation_vector = np.array([self.max_translation_x, self.max_translation_y,0])
        self.human_saccade_time = 0.1
        self.max_observation_store = 5
        self.time_step_size = 0.005
        self._max_episode_steps = ((1/self.time_step_size)*20)/self.steps_per_action # 20 seconds
        self.max_in_glas = 250
        self.max_particles = count_particles(self.fluid_base_path)
        self.max_water_flow = 4
        self.fix_step = (self.time_step_size/self.human_saccade_time)*self.steps_per_action

        self.reset(first=True)

    def seed(self,seed):
        np.random.seed(seed)

    def _walk_uncertaintys(self):
        if self.uncertainty == 0:
            return
        gaussian = norm(0,self.uncertainty).pdf
        for key in self.current_walk:
            proposal_accepted = False
            while not proposal_accepted:
                proposal = np.random.normal(self.current_walk[key],self.uncertainty*self.proposal_function_rate)
                cur_height = gaussian(self.current_walk[key])
                if random.random() < gaussian(proposal)/cur_height:
                    proposal_accepted = True
            self.current_walk[key] = proposal

    def reset(self,first=False):
        if self.uncertainty == 0:
            remove_particles(self.fluid_base_path,os.path.join(FILE_PATH,"models/bigger/tmp_fluid.bgeo"),1)
        else:
            remove_particles(self.fluid_base_path,os.path.join(FILE_PATH,"models/bigger/tmp_fluid.bgeo"),0.5+random.random()/2)
        
        if not first:
            self.base.cleanup()
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
        self.base.initSimulation()
        self.base.initBoundaryData()

        self.bottle = Model3d(self.sim.getCurrent().getBoundaryModel(1).getRigidBodyObject())
        self.glas = Model3d(self.sim.getCurrent().getBoundaryModel(0).getRigidBodyObject())
        self.fluid = self.sim.getCurrent().getFluidModel(0)
        #print("particle count:",self.fluid.numActiveParticles())

        self.particle_locations = self._get_particle_locations()

        self.fill_observations = deque(maxlen=self.max_observation_store)
        self.water_flow_observations = deque(maxlen=self.max_observation_store)
        self.fixation = 0
        self.fixation_direction = 0
        if self.uncertainty == 0:
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
                "rotation":np.random.normal(0,self.uncertainty),
                "translation_x":np.random.normal(0,self.uncertainty),
                "translation_y":np.random.normal(0,self.uncertainty),
                "fill_estimate":np.random.normal(0,self.uncertainty),
                "water_flow":np.random.normal(0,self.uncertainty),
                "bottle_fill":np.random.normal(0,self.uncertainty)
            }
        self.time = 0
        self._step_number = 0
        return self._observe()

    def _get_reward(self):
        def score_locations(locations):
            return locations["glas"]-locations["spilled"]*5
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
                fixation/saccade location, bottle_fill, estimated fluid level in glas, estimated rate of water flow.
            reward (float) :
                +1 for landing a water particle in the glas. -10 for spilling a water particle.
            episode_over (bool) :
                Whether it's time to reset the environment again.
            info (dict) :
                Some particle statistics
        """
        action = np.array(action)
        punish = 0
        if len(action)!=4:
            raise ValueError(f"Invalid action {action}")
        for i,a in enumerate(action):
            if a<0:
                punish -= a
                action[i] = 0
            elif a>1:
                punish += a-1
                action[i] = 1
        self._step_number += 1
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
        if self.bottle.translation[0] + to_translate[0]*self.steps_per_action > self.translation_bounds[0][1]:
            to_translate[0] = (self.translation_bounds[0][1] - self.bottle.translation[0])/self.steps_per_action
        if self.bottle.translation[0] + to_translate[0]*self.steps_per_action < self.translation_bounds[0][0]:
            to_translate[0] = (self.translation_bounds[0][0] - self.bottle.translation[0])/self.steps_per_action
        if self.bottle.translation[1] + to_translate[1]*self.steps_per_action > self.translation_bounds[1][1]:
            to_translate[1] = (self.translation_bounds[1][1] - self.bottle.translation[1])/self.steps_per_action
        if self.bottle.translation[1] + to_translate[1]*self.steps_per_action < self.translation_bounds[1][0]:
            to_translate[1] = (self.translation_bounds[1][0] - self.bottle.translation[1])/self.steps_per_action
        to_rotate = R.from_euler("z",rot_radians).as_matrix()
        for step in range(self.steps_per_action):
            self.bottle.rotate(to_rotate)
            self.bottle.translate(to_translate)
            self.time += self.time_step_size
            self.base.timeStepNoGUI()
        self.bottle.body.updateVertices()
        old_particle_locations = self.particle_locations
        reward = self._get_reward()-punish
        if self.fixation in (0,1):
            self.fixation_direction = -1 if action[3] < 0.5 else 1
        old_fix = self.fixation
        self.fixation += self.fixation_direction*self.fix_step
        self.fixation = 0 if self.fixation<0 else (1 if self.fixation>1 else self.fixation)
        if self.fixation == 0:
            if old_fix!=0:
                self.fill_observations.clear()
            self.fill_observations.append(self.particle_locations["glas"])
        elif self.fixation == 1:
            if old_fix!=1:
                self.water_flow_observations.clear()
            self.water_flow_observations.append(old_particle_locations["bottle"]-self.particle_locations["bottle"])
        self._walk_uncertaintys()
        observation = self._observe()
        return observation,reward,self._step_number>self._max_episode_steps or self.particle_locations["spilled"]>=10,self.particle_locations

    def render(self, mode='human'):
        if self.gui is None:
            raise Exception("Trying to render without initializing with gui_mode=True")
        self.gui.one_render()

    def _normalize_observation(self,rotation,translation_x,translation_y,fixation,in_bottle,fill_state,water_flow_estimate):
        rotation = (rotation + math.pi)/(2*math.pi)
        translation_x = (translation_x - self.translation_bounds[0][0]) / (self.translation_bounds[0][1]-self.translation_bounds[0][0])
        translation_y = (translation_x - self.translation_bounds[1][0]) / (self.translation_bounds[1][1]-self.translation_bounds[1][0])
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
        return rotation,translation_x,translation_y,fixation,in_bottle,fill_state,water_flow_estimate


    def _observe(self):
        rotation = R.from_matrix(self.bottle.rotation).as_euler("zyx")[0]
        translation_x,translation_y = self.bottle.translation[:2]
        water_flow_estimate = np.mean(self.water_flow_observations) if len(self.water_flow_observations)>0 else 0
        fill_state = np.mean(self.fill_observations) if len(self.fill_observations)>0 else 0
        in_bottle = self.particle_locations["bottle"]
        return self._normalize_observation(rotation,translation_x,translation_y,self.fixation,in_bottle,fill_state,water_flow_estimate)

if __name__ == "__main__":
    env = Pouring_env(gui_mode=True)
    for _ in range(1000):
        obs,reward,done,info = env._step((0,0.5,0.5,0))
        env._render()
        print(obs,reward,done,info)

