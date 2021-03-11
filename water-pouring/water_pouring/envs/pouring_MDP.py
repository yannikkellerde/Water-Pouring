from pouring_base import Pouring_base
from gym import spaces
from scipy.spatial.transform import Rotation as R
import math
import numpy as np
import os,sys

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
class Pouring_MDP(Pouring_base):
    """A concrete water-pouring gym environment that uses the full state as
    observations so that it qualifies as a Markov Decision Process.
    """
    def __init__(self,**kwargs):
        """Initialize pouring-MDP and set action and observation space.
        Args:
            **kwargs: Keyword arguments that are forwarded to the abstract init method
                      of the base implementation.
        """
        super(Pouring_MDP, self).__init__(**kwargs)
        self.action_space = spaces.Box(low=-1,high=1,shape=(3,))
        self.observation_space = spaces.Tuple((spaces.Box(low=-1,high=1,shape=(7+(2*self.action_space.shape[0] if self.jerk_punish>0 else 0),)),
                                               spaces.Box(low=-1,high=1,shape=(self.max_particles,9))))

    def _to_observations(self,tsp,spill_punish,target_fill):
        """Normalize parameter values to be beween -1 and 1 so that they
        can be used as observations.

        Args:
            tsp: Negative reward per timestep ("time_step_punish").
            spill_punish: Negative reward per spilled particle.
            target_fill: Target fill-level of the glass, the agent is supposed to reach.
        Returns:
            A 3 tuple that contains the argument values normalized into a range of -1 to 1
        """
        tsp_obs = ((tsp-self.time_step_punish_range[0]) /
                   (self.time_step_punish_range[1]-self.time_step_punish_range[0]))*2-1
        spill_punish_obs = ((spill_punish-self.spill_range[0]) /
                            (self.spill_range[1]-self.spill_range[0]))*2-1
        target_fill_obs = ((target_fill-self.target_fill_range[0]) /
                            (self.target_fill_range[1]-self.target_fill_range[0]))*2-1
        return tsp_obs,spill_punish_obs,target_fill_obs

    def _observe(self):
        """Observe the full state of the environment.

        Returns:
            A 2-tuple containing:
            1) Feature data, a 7 or 13 dimensional array that includes:
                1. Bottle Rotation
                2. The x-translation of the bottle
                3. The y-translation of the bottle
                4. This episodes time_step_punish
                5. This episodes spill_punish
                6. This episodes target_fill_state
                7. The number of steps that have been performed since the start of the episode.
                8-10. If self.jerk_punish > 0, the last performed action
                11-14. If self.jerk_punish > 0, the next to last performed action
            2) Particle data, a num_particles X 9 dimensional Matrix with each row containing:
                1-3. The 3d position of the water particle.
                4-6. The 3d velocity of the water particle.
                7. The bottle rotation.
                8. The bottle x-translation.
                9. The bottle y-translation.

            All values in both tuple entries are normalized to the range -1 to 1.
        """
        fluid_data = []
        rotation = R.from_matrix(self.bottle.rotation).as_euler("zyx")[0]
        rotation = (rotation-self.min_rotation)/(math.pi-self.min_rotation)
        translation_x,translation_y = self.bottle.translation[:2]
        translation_x = (translation_x - self.translation_bounds[0][0]) / (self.translation_bounds[0][1]-self.translation_bounds[0][0])
        translation_y = (translation_y - self.translation_bounds[1][0]) / (self.translation_bounds[1][1]-self.translation_bounds[1][0])

        for i in range(self.fluid.numActiveParticles()):
            # Iterate above all fluid particles in the simulator and get their position/velocity
            pos = self.fluid.getPosition(i)
            vel = self.fluid.getVelocity(i)
            fluid_data.append((pos[0],pos[1],pos[2],vel[0],vel[1],vel[2],rotation,translation_x,translation_y))
        fluid_data = np.clip(fluid_data,-1,1)
        
        time_obs = (self._step_number/self._max_episode_steps)*2-1
        tsp_obs,spill_punish_obs,target_fill_obs = self._to_observations(self.time_step_punish,self.spill_punish,self.target_fill_state)

        feat_dat = [rotation,translation_x,translation_y,tsp_obs,spill_punish_obs,target_fill_obs,time_obs]
        if self.jerk_punish>0:
            feat_dat.extend(np.array(self.last_actions)[:-1].flatten())
        feat_dat = np.array(feat_dat)
        return feat_dat,fluid_data