from pouring_base import Pouring_base
from gym import spaces
from scipy.spatial.transform import Rotation as R
from collections import deque
import math
import numpy as np
import os,sys

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
class Pouring_featured(Pouring_base):
    """A concrete water-pouring gym environment that uses handcrafted features as
    observations of the state. Thus, this environment describes a Partially Observable
    Markov Decision Process.

    Attributes:
        max_in_air: Maximum amount of water-particles in the air that is assumed to be
                    possible. Used for normalization of observations.
    """
    def __init__(self,**kwargs):
        """Initialize the water-pouring environment.

        Args:
            **kwargs: Keyword arguments that are forwarded to the abstract init method
                      of the base implementation.
        """
        self.max_in_air = 40
        super(Pouring_featured, self).__init__(**kwargs)
        self.action_space = spaces.Box(low=-1,high=1,shape=(3,))
        self.observation_space = spaces.Box(low=-1,high=1,shape=(11+(2*self.action_space.shape[0] if self.jerk_punish>0 else 0),))

    def _observe(self):
        """Make an observation of the current state by the use of handcrafted features, which
        do not describe the full state completely.

        Returns:
            A 11 or 17 dimensional numpy array that contains:
                1. Bottle Rotation
                2. The x-translation of the bottle
                3. The y-translation of the bottle
                4. This episodes time_step_punish
                5. This episodes spill_punish
                6. This episodes target_fill_state
                7. The number of steps that have been performed since the start of the episode.
                8. The fill-level of the glass.
                9. The amount of water in the bottle.
                10. The amount of water in the air between bottle and glass.
                11. The amount of spilled particles.
                12-14. If self.jerk_punish > 0, the last performed action.
                15-17.  If self.jerk_punish > 0, the next to last performed action
            
            All values in the array are normalized to the range -1 to 1.
        """
        rotation = R.from_matrix(self.bottle.rotation).as_euler("zyx")[0]
        rotation = (rotation-self.min_rotation)/(math.pi-self.min_rotation)
        translation_x,translation_y = self.bottle.translation[:2]
        translation_x = (translation_x - self.translation_bounds[0][0]) / (self.translation_bounds[0][1]-self.translation_bounds[0][0])
        translation_y = (translation_y - self.translation_bounds[1][0]) / (self.translation_bounds[1][1]-self.translation_bounds[1][0])

        tsp_obs = ((self.time_step_punish-self.time_step_punish_range[0]) /
                   (self.time_step_punish_range[1]-self.time_step_punish_range[0]))*2-1
        time_obs = (self._step_number/self._max_episode_steps)*2-1
        spill_punish_obs = ((self.spill_punish-self.spill_range[0]) /
                            (self.spill_range[1]-self.spill_range[0]))*2-1
        target_fill_obs = ((self.target_fill_state-self.target_fill_range[0]) /
                            (self.target_fill_range[1]-self.target_fill_range[0]))*2-1
        feat_dat = [rotation,translation_x,translation_y,tsp_obs,spill_punish_obs,target_fill_obs,time_obs]
        feat_dat.append((self.particle_locations["glass"]/self.max_in_glass)*2-1)
        feat_dat.append((self.particle_locations["bottle"]/self.max_particles)*2-1)
        feat_dat.append((self.particle_locations["air"]/self.max_in_air)*2-1)
        feat_dat.append((self.particle_locations["spilled"]/self.max_spill)*2-1)
        if self.jerk_punish>0:
            feat_dat.extend(np.array(self.last_actions)[:-1].flatten())
        feat_dat = np.clip(np.array(feat_dat),-1,1)
        return feat_dat
