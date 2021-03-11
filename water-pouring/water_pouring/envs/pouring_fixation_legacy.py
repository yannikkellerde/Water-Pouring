from pouring_featured import Pouring_featured
from gym import spaces
from scipy.spatial.transform import Rotation as R
from collections import deque
import math
import numpy as np
import os,sys

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
class Pouring_fixation(Pouring_featured):
    """A concrete water-pouring gym environment that uses handcrafted features as
    observations of the state. Thus, this environment describes a Partially Observable
    Markov Decision Process.

    Attributes:
        max_in_air: Maximum amount of water-particles in the air that is assumed to be
                    possible. Used for normalization of observations.
    """
    def __init__(self,saccade_time=0,**kwargs):
        """Initialize the water-pouring environment.

        Args:
            **kwargs: Keyword arguments that are forwarded to the abstract init method
                      of the base implementation.
        """
        self.fix_list = []
        self.cur_fixation = 1
        super(Pouring_fixation, self).__init__(**kwargs)
        self.action_space = spaces.Box(low=-1,high=1,shape=(4,))
        if saccade_time == 0:
            self.saccade_speed = 2
        else:
            self.saccade_speed = np.clip(((self.time_step_size*self.steps_per_action)/saccade_time)*2,-2,2)
        self.observation_space = spaces.Box(low=-1,high=1,shape=(12+(2*self.action_space.shape[0] if self.jerk_punish>0 else 0),))

    def reset(self,**kwargs):
        if len(self.fix_list)>0:
            print(f"Percent fixations on water jet: {sum(self.fix_list)/(len(self.fix_list)*2)+0.5}")
            self.fix_list = []
        return super(Pouring_fixation,self).reset(**kwargs)

    def step(self,action):
        if len(action)<4:
            raise ValueError(f"Invalid action {action}")
        if action[3] > 0:
            self.cur_fixation+=self.saccade_speed
        else:
            self.cur_fixation-=self.saccade_speed
        self.cur_fixation = np.clip(self.cur_fixation,-1,1)
        self.fix_list.append(self.cur_fixation)
        return super(Pouring_fixation,self).step(action[:3])

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
                8. The fill-level of the glass
                9. The amount of water in the bottle.
                10. The amount of water in the air between bottle and glass.
                11. The amount of spilled particles.
                12. The current fixation (fill-level or particles in air).
                13-15. If self.jerk_punish > 0, the last performed action.
                16-18.  If self.jerk_punish > 0, the next to last performed action
            
            All values in the array are normalized to the range -1 to 1.
        """
        feat_dat = super(Pouring_fixation,self)._observe()
        if self.cur_fixation == 1:
            feat_dat[7] = 0
        else:
            feat_dat[9] = 0
        feat_dat = list(feat_dat)
        feat_dat.insert(11,self.cur_fixation)
        return np.array(feat_dat)
