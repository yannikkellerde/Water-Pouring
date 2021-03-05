from pouring_base import Pouring_base
from pouring_MDP import Pouring_MDP
from gym import spaces
from scipy.spatial.transform import Rotation as R
import math
import numpy as np
import os,sys

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
class Pouring_G2G_MDP(Pouring_MDP):
    """Concrete glass to glass water-pouring gym environment that uses the full state as
    observations and is thus a Markov Descision Process."""
    def __init__(self,**kwargs):
        """Initialize the glass to glass water pouring environment.

        Args:
            **kwargs: Keyword arguments that are forwarded to the abstract init method
                      of the base implementation.
        """
        super(Pouring_G2G_MDP, self).__init__(scene_base=os.path.join(FILE_PATH,"scenes","glass_to_glass.json"),**kwargs)
        self.min_rotation = 0
        self.max_rotation_radians = self.max_rotation_radians*2
        self.base_translation_vector*=2
        self.max_in_glass = self.max_particles
        self.target_fill_range = [30,self.max_in_glass]
        self.target_fill_state = self.max_in_glass
        self.translation_bounds = ((-0.1,0.3),(-0.04,0.3))
        self.observation_space = spaces.Tuple((spaces.Box(low=-1,high=1,shape=(7,)),
                                               spaces.Box(low=-1,high=1,shape=(self.max_particles,9))))
        self.reset(first=True)