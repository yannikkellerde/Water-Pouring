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
        super(Pouring_mdp_full, self).__init__(**kwargs)
        self.action_space = spaces.Box(low=-1,high=1,shape=(3,))
        self.particle_list_length = 3
        self.observation_space = spaces.Box(low=-1,high=1,shape=(5+4*self.particle_list_length,))
        self.max_in_air = 10

    def reset(self,*args,**kwargs):
        self.paricle_hist = deque(maxlen=self.particle_list_length)
        super(Pouring_featured, self).reset(*args,**kwargs)

    def _particle_hist_to_observation(self):
        observes = []
        for i in range(self.particle_locations):
            if len(self.particle_hist) > i:
                obs = []
                obs.append(self.particle_hist[i]["glas"]/self.max_in_glas)
                obs.append(self.particle_hist[i]["bottle"]/self.max_particles)
                obs.append(self.particle_hist[i]["air"]/self.max_in_air)
                obs.append(self.particle_hist[i]["spilled"]/self.max_spill)
                observes.append(obs)
            else:
                observes.append(observes[-1])
        return sum(observes,[])

    def _observe(self):
        self.particle_hist.appendleft(self.particle_locations)
        rotation = R.from_matrix(self.bottle.rotation).as_euler("zyx")[0]
        rotation = (rotation-self.min_rotation)/(math.pi-self.min_rotation)
        translation_x,translation_y = self.bottle.translation[:2]
        translation_x = (translation_x - self.translation_bounds[0][0]) / (self.translation_bounds[0][1]-self.translation_bounds[0][0])
        translation_y = (translation_y - self.translation_bounds[1][0]) / (self.translation_bounds[1][1]-self.translation_bounds[1][0])

        tsp_obs = ((self.time_step_punish-self.time_step_punish_range[0]) /
                   (self.time_step_punish_range[1]-self.time_step_punish_range[0]))*2-1
        time_obs = (self._step_number/self._max_episode_steps)*2-1
        feat_dat = [rotation,translation_x,translation_y,tsp_obs,time_obs]
        feat_dat.extend(self._particle_hist_to_observation())
        feat_dat = np.clip(np.array(feat_dat),-1,1)
        return feat_dat
