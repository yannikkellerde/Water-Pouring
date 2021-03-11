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
from abc import ABC, abstractmethod

class Pouring_base(ABC,gym.Env):
    """An abstract base class for a water-pouring gym environment that uses
    SPlishSPlash as a fluid simulator.

    Attributes:
        action_space: Gym action space that defines the shape of possible actions that can be performed.
        observation_space: Gym observation space that defines the shape of observations obtained from the
                           environment.
        policy_uncertainty: Standard deviation of the signal dependent noise added to
                            all actions from the agent.
        translation_bounds: Maximimum and minimum bottle translation in x and y direction.
        base_translation_vector: 3d Vector capturing the maximum amount of meters the bottle can be translated
                                 in x,y or z direction.
        max_rotation_radians: Maximum amount of radians, the bottle can be rotated each step.
        min_rotation: If the bottle is lowered below this amount of radians, the episode ends (assuming at
                      least on water-particle has been successfully poured in)
        time_step_punish_range: The range in which the time_step_punish varies when it is randomized.
        spill_range: The range in which the spill_punish varies when it is randomized.
        target_fill_range: The range in which the target_fill_state varies when it is randomized.
        time_step_punish: Negative reward per step taken in the environment.
        spill_punish: Negative reward given per spilled water-particle.
        target_fill_state: Target fill-level the agent is supposed to reach.
        fixed_tsp: If true, the time_step_punish won't be randomized when resetting the environment.
        fixed_spill: If true, the spill_punish won't be randomized when resetting the environment.
        fixed_target_fill: If true, the target_fill_state won't be randomized when resetting the environment.
        hit_reward: Reward that is gained per correctly poured in particle.
        max_spill: After this many spilled partilces, the episode ends.
        jerk_punish: Scaling factor for the jerk-based reward regularization that punishes non-smooth motion.
        use_gui: If true, rendering is enables. Otherwise trying to render will throw a RuntimeException.
        scene_file: Path to the SPlisHSPlasH scene file that describes the setting of the pouring scene.
        steps_per_action: One step in the gym environment corresponds to this many steps in the fluid
                          simulator.
        time_step_size: One step in the fluid simulator captures this many seconds of simulation.
        _max_episode_steps: After this many steps an episode ends.
        max_in_glass: Volume of the glass in number of water-particles that fit.
        max_particles: Total number of water-particles in the simulation.
        gui: Reference to a SPlisHSPlasH gui.
        sim: Reference to the SPlisHSPlasH Simulator instance.
        base: Reference to the SPlisHSPlasH SimulatorBase instance.
        bottle: A Model3d instance that handles the properties of the bottle.
        glass: A Model3d instance that handles the properties of the glass.
        fluid: A reference to the SPlisHSPlasH fluid model.
        particle_locations: A dictionary obtained from _get_particle_locations() that stores where
                            water-particles are in the environment.
        time: Simulation time in seconds that has passed since the start of the episode.
        _step_number: How many steps have been performed since the start of the episode.
        done: Weather it is time to reset the environment.
        last_actions: The actions performed in the last two steps.
    """
    metadata = {'render.modes': ['human']} # Gym compatibility

    @abstractmethod
    def __init__(self, use_gui=False, fixed_spill=True, fixed_tsp=True,
                 fixed_target_fill=True,policy_uncertainty=0,jerk_punish=0,
                 scene_base=os.path.join("scenes","simple_scene.json")):
        """
        Initializes the gym environment as well as the fluid simulator.

        Args:
            use_gui: Initialize the simulator for rendering. Without use_gui=True,
                     rendering throws an error.
            fixed_spill: When fixed_spill=False, the spill_punish will be randomized
                        every time the environment is reset.
            fixed_tsp: When fixed_tsp=False, the time_step_punish will be randomized
                        every time the environment is reset.
            fixed_target_fill: When fixed_target_fill=False, the target_fill_state will
                                be randomized every time the environment is reset.
            policy_uncertainty: Standard deviation of the signal dependent noise added to
                                all actions from the agent.
            jerk_punish: Scaling factor for how much negative reward is given for very
                        non-smooth actions (High third derrivative magnitude (jerk))
            scene_base: The scene file that will used to set the environment
        """

        self.action_space = NotImplemented
        self.observation_space = NotImplemented

        # Actions and Movement
        self.policy_uncertainty = policy_uncertainty
        self.translation_bounds = ((-0.1,0.2),(-0.04,0.3))
        max_translation_x = 4e-4
        max_translation_y = 4e-4
        self.base_translation_vector = np.array([max_translation_x, max_translation_y,0])
        self.max_rotation_radians = 0.003
        self.min_rotation = 1.22

        # Rewards
        self.max_in_glass = 390
        self.time_step_punish_range = [0,2]
        self.spill_range = [1,50]
        self.target_fill_range = [30,self.max_in_glass]
        self.fixed_tsp = fixed_tsp
        self.fixed_spill = fixed_spill
        self.fixed_target_fill = fixed_target_fill
        self.hit_reward = 1
        self.time_step_punish = 1
        self.spill_punish = 15
        self.max_spill = 20
        self.target_fill_state = self.max_in_glass
        self.jerk_punish = jerk_punish


        # Simulation
        self.use_gui = use_gui
        self.scene_file = os.path.join(FILE_PATH,scene_base)
        self.steps_per_action = 20
        self.time_step_size = util.extract_time_step_size(self.scene_file)
        self._max_episode_steps = ((1/self.time_step_size)*20)/self.steps_per_action # 20 seconds
        fluid_base_path = os.path.join(os.path.dirname(self.scene_file),util.get_fluid_path(self.scene_file))
        self.max_particles = count_particles(fluid_base_path)
        self.gui = None

        self.reset(first=True)

    def seed(self,seed):
        """
        Seeding might not work perfectly for this environment
        because the simulator does not behave exactly the same
        every time.

        Args:
            seed: An integer seed for the random number generator.
        """
        np.random.seed(seed)


    def reset(self,first=False,use_gui=False,printstate=False):
        """
        Reset the gym environment as well as the fluid simulator.

        Args:
            first: Only set to true when calling from the constructor.
            use_gui: If true, the environment will be initialized for rendering.
            printstate: If true, print the new values of randomized parameters and
                        the final particle states from the last episode.

        Returns:
            The state of the environment after resetting
        """
        if not first:
            self.base.cleanup()
        if self.gui is not None:
            self.gui.die()
        self.gui = None
        if use_gui:
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
        self.glass = Model3d(self.sim.getCurrent().getBoundaryModel(0).getRigidBodyObject(),stretch_vertices=0.1)
        self.fluid = self.sim.getCurrent().getFluidModel(0)

        if not self.fixed_tsp:
            self.time_step_punish = util.get_random(self.time_step_punish_range)
        if not self.fixed_spill:
            self.spill_punish = util.get_random(self.spill_range)
        if not self.fixed_target_fill:
            self.target_fill_state = util.get_random(self.target_fill_range)

        if printstate:
            if not first:
                print("In glass:",self.particle_locations["glass"])
                print("Spilled:",self.particle_locations["spilled"])
            print("Spill punish",self.spill_punish)
            print("TSP",self.time_step_punish)
            print("Target fill",self.target_fill_state)

        self.particle_locations = self._get_particle_locations()
        self.time = 0
        self._step_number = 0
        self.done = False
        self.last_actions = deque([[0,0,0] for i in range(3)],maxlen=3)
        return self._observe()


    def _score_locations(self,locations,target_fill_state,spill_punish,max_spill):
        """
        Calculate the a score for how good the current state is based on the locations
        of all water particles.

        Args:
            locations: A dictionary that has been obtained by calling _get_particle_locations
            target_fill_state: The fill-level the agent is supposed to reach. This may be different
                               from the attribute self.target_fill_level
            spill_punish: How negatively should spilled particles influence the score
            max_spill: Describes after how much spillage an episode does end.

        Returns:
            A two entry numpy array containing a score for any positive things about the state
            and a score for everything negative.
        """
        hit_reward = (self.max_in_glass/target_fill_state)*self.hit_reward
        return np.array((target_fill_state-abs(locations["glass"]-target_fill_state)*hit_reward,min(locations["spilled"],max_spill)*spill_punish))


    def _imagine_reward(self,time_step_punish,spill_punish,target_fill_state):
        """Imagine what rewards would be given if the parameters were different.

        Args:
            time_step_punish: Default negative reward for each step taken in the environment.
            spill_punish: Negative reward per spilled particle.
            target_fill_state: The target fill-level the agent is supposed to reach.

        Returns:
            A scalar reward calcuated for the given parameters.
        """
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
        """Calculate the amount of reward that is obtained by the transition to the current state.

        Returns:
            A scalar reward value.
        """
        return self._imagine_reward(self.time_step_punish,self.spill_punish,self.target_fill_state)


    def _get_particle_locations(self):
        """Figure out where the water particles are in the current state. This means answering the
        questions:
            1) How many particles are in the bottle
            2) How many particles are in the glass
            3) How many particles are in the air between the bottle and the glass
            4) How many particles have been spilled

        Returns:
            A dictionary that maps from 'bottle', 'glass', 'air' and 'spilled' to the integer number
            of water particles in that location.
        """
        res = {
            "bottle":0,
            "glass":0,
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
        in_glass = self.glass.hull.find_simplex(fluid_positions)>=0
        res["glass"] = np.sum(in_glass)
        fluid_positions = fluid_positions[~in_glass]
        spilled = fluid_positions[:,1]<self.glass.orig_rect[1][1]
        res["spilled"] = np.sum(spilled)
        fluid_positions = fluid_positions[~spilled]
        res["air"] = len(fluid_positions)
        return res


    def step(self,action):
        """Perform one step in the environment using the fluid simulator.

        Parameters:
        action: A 3-dimensional vector of floats between -1 and 1 representing:
                1. Rotation: For values < 0 rotate the botte downwards, for > 0 upwards.
                2. Translation-x: For < 0 move bottle towards glass, for > 0 away from glass.
                3. Translation-y: For < 0 move bottle down, for > 0, move up.

        Returns:
            A four-tuple consisting of:
            1) observation:
                The observation of the environment. Shape depends on concrete value of
                self.observation_space
            2) reward:
                The reward obtained by performing the action in the environment
            3) done:
                Whether it's time to reset the environment again.
            4) info:
                An empty dictionary.
        Raises:
            ValueError: If an invalid action was given as an argument.
        """
        action = np.array(action)

        # Store performed action to potentially output as additional observations.
        self.last_actions.appendleft(action)

        if len(action)!=3 or np.any(action>1) or np.any(action<-1):
            raise ValueError("Invalid action {}".format(action))

        if self.policy_uncertainty>0:  # Add signal dependent noise
            action = np.random.normal(action,np.abs(action)*self.policy_uncertainty)
        self._step_number += 1

        # Tranlate performed action into actual performed rotation and translation.
        rot_radians = -action[0]*self.max_rotation_radians
        to_translate = self.base_translation_vector*np.array([action[1],action[2],0],dtype=np.float)

        # Do not move the bottle out of translation bounds
        if (self.bottle.translation[0] + to_translate[0]*self.steps_per_action > self.translation_bounds[0][1] or
                self.bottle.translation[0] + to_translate[0]*self.steps_per_action < self.translation_bounds[0][0]):
            to_translate[0] = 0
        if (self.bottle.translation[1] + to_translate[1]*self.steps_per_action > self.translation_bounds[1][1] or
                self.bottle.translation[1] + to_translate[1]*self.steps_per_action < self.translation_bounds[1][0]):
            to_translate[1] = 0

        # Episode end conditions

        # Bottle angle lowered below threshold
        if ((R.from_matrix(self.bottle.rotation).as_euler("zyx")[0]<self.min_rotation) and self.particle_locations["air"]==0 and self.particle_locations["glass"]!=0):
            self.done = True

        # Maximum number of steps reached
        if (self._step_number>self._max_episode_steps):
            self.done = True

        # Too many spilled partilces
        if (self.particle_locations["spilled"]>=self.max_spill):
            self.done = True

        # Prevent bottle glass collision
        tt,rr = self.glass.check_if_in_rect(self.bottle,to_translate*self.steps_per_action,rot_radians*self.steps_per_action)

        # Ajust rotation/translation magnitude for a single step in the fluid simulator
        to_translate = tt/self.steps_per_action
        rot_radians = rr/self.steps_per_action

        to_rotate = R.from_euler("z",rot_radians).as_matrix()
        for step in range(self.steps_per_action):
            # Actually rotate and translate the bottle in the simulator
            self.bottle.rotate(to_rotate)
            self.bottle.translate(to_translate)
            self.time += self.time_step_size
            self.base.timeStepNoGUI() # Perform single time step in fluid simulator
        self.bottle.body.updateVertices()
        reward = self._get_reward()
        observation = self._observe()
        return observation,reward,self.done,{}


    def render(self, mode='human'):
        """Render the environment using a gui window.
        Args:
            mode: Does nothing but exists for compatibility with gym.
        Raises:
            RuntimeError: When the environment was not initialized or reset with use_gui=True
        """
        if self.gui is None:
            raise RuntimeError("Trying to render without initializing with use_gui=True")
        self.gui.one_render()


    @abstractmethod
    def _observe(self):
        """Observe the current state of the environment.
        """
        pass