from pouring_base import Pouring_base
from pouring_featured import Pouring_featured
from gym import spaces
from scipy.spatial.transform import Rotation as R
import math
import numpy as np
import os,sys

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
class Pouring_G2G_featured(Pouring_featured):
    def __init__(self,**kwargs):
        super(Pouring_G2G_featured, self).__init__(scene_base=os.path.join(FILE_PATH,"scenes","glass_to_glass.json"),**kwargs)
        self.min_rotation = 0
        self.max_rotation_radians = 0.006
        self.max_translation_x = 0.003
        self.max_translation_y = 0.003
        self.base_translation_vector = np.array([self.max_translation_x, self.max_translation_y,0])
        self.max_in_glas = 187
        self.max_in_air = 80
        self.translation_bounds = ((-0.5,1.5),(-0.2,1.5))