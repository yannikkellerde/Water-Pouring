from pouring_base import Pouring_base
from pouring_mdp_full import Pouring_mdp_full
from gym import spaces
from scipy.spatial.transform import Rotation as R
import math
import numpy as np
import os,sys

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
class Pouring_G2G_mdp(Pouring_mdp_full):
    def __init__(self,**kwargs):
        super(Pouring_G2G_mdp, self).__init__(scene_base=os.path.join(FILE_PATH,"scenes","glass_to_glass.json"),**kwargs)
        #self.action_space = spaces.Box(low=-1,high=1,shape=(3,))
        #self.observation_space = spaces.Tuple((spaces.Box(low=-1,high=1,shape=(6,)),
        #                                       spaces.Box(low=-1,high=1,shape=(self.max_particles,9))))
        self.min_rotation = 0
        #self.max_rotation_radians = self.max_rotation_radians*2
        #self.max_translation_x = self.max_translation_x*2
        #self.max_translation_y = self.max_translation_y*2
        self.base_translation_vector = np.array([self.max_translation_x, self.max_translation_y,0])
        self.max_in_glas = self.max_particles
        self.target_fill_range = [30,self.max_in_glas]
        self.target_fill_state = self.max_in_glas
        self.translation_bounds = ((-0.5,1.5),(-0.2,1.5))
        self.observation_space = spaces.Tuple((spaces.Box(low=-1,high=1,shape=(7,)),
                                               spaces.Box(low=-1,high=1,shape=(self.max_particles,9))))
        self.reset(first=True)