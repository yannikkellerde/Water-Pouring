from pouring_base import Pouring_base
from gym import spaces
from scipy.spatial.transform import Rotation as R
from collections import deque
import math
import numpy as np
import os,sys

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
class Pouring_featured(Pouring_base):
    def __init__(self,**kwargs):
        self.max_in_air = 40
        super(Pouring_featured, self).__init__(**kwargs)
        self.action_space = spaces.Box(low=-1,high=1,shape=(3,))
        self.observation_space = spaces.Box(low=-1,high=1,shape=(11,))

    def _observe(self):
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
        feat_dat.append((self.particle_locations["glas"]/self.max_in_glas)*2-1)
        feat_dat.append((self.particle_locations["bottle"]/self.max_particles)*2-1)
        feat_dat.append((self.particle_locations["air"]/self.max_in_air)*2-1)
        feat_dat.append((self.particle_locations["spilled"]/self.max_spill)*2-1)
        feat_dat = np.clip(np.array(feat_dat),-1,1)
        return feat_dat
