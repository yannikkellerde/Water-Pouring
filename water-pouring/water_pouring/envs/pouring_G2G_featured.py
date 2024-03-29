from pouring_base import Pouring_base
from pouring_featured import Pouring_featured
from gym import spaces
from scipy.spatial.transform import Rotation as R
import math
import numpy as np
import os,sys

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
class Pouring_G2G_featured(Pouring_featured):
    """Concrete glass to glass water-pouring gym environment that uses the handcrafted features as
    observations which do not fully describe the full state. The environment thus implements a
    Partially Observable Markov Decision Process.
    """
    def __init__(self,**kwargs):
        """Initialize the glass to glass water pouring environment.

        Args:
            **kwargs: Keyword arguments that are forwarded to the abstract init method
                      of the base implementation.
        """
        super(Pouring_G2G_featured, self).__init__(scene_base=os.path.join(FILE_PATH,"scenes","glass_to_glass.json"),**kwargs)
        self.min_rotation = 0
        self.max_rotation_radians = self.max_rotation_radians*2
        self.base_translation_vector*=2
        self.max_in_glass = self.max_particles
        self.target_fill_range = [30,self.max_in_glass]
        self.target_fill_state = self.max_in_glass
        self.translation_bounds = ((-0.1,0.3),(-0.04,0.3))
        self.reset(first=True)