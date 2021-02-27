from pouring_base import Pouring_base
from gym import spaces
from scipy.spatial.transform import Rotation as R
import math
import numpy as np
import os,sys

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
class Pouring_mdp_full(Pouring_base):
    def __init__(self,**kwargs):
        super(Pouring_mdp_full, self).__init__(**kwargs)
        self.action_space = spaces.Box(low=-1,high=1,shape=(3,))
        self.observation_space = spaces.Tuple((spaces.Box(low=-1,high=1,shape=(7+(3*self.action_space.shape[0] if self.jerk_punish>0 else 0),)),
                                               spaces.Box(low=-1,high=1,shape=(self.max_particles,9))))

    def _to_observations(self,tsp,spill_punish,target_fill):
        tsp_obs = ((tsp-self.time_step_punish_range[0]) /
                   (self.time_step_punish_range[1]-self.time_step_punish_range[0]))*2-1
        spill_punish_obs = ((spill_punish-self.spill_range[0]) /
                            (self.spill_range[1]-self.spill_range[0]))*2-1
        target_fill_obs = ((target_fill-self.target_fill_range[0]) /
                            (self.target_fill_range[1]-self.target_fill_range[0]))*2-1
        return tsp_obs,spill_punish_obs,target_fill_obs

    def _observe(self):
        fluid_data = []
        rotation = R.from_matrix(self.bottle.rotation).as_euler("zyx")[0]
        rotation = (rotation-self.min_rotation)/(math.pi-self.min_rotation)
        translation_x,translation_y = self.bottle.translation[:2]
        translation_x = (translation_x - self.translation_bounds[0][0]) / (self.translation_bounds[0][1]-self.translation_bounds[0][0])
        translation_y = (translation_y - self.translation_bounds[1][0]) / (self.translation_bounds[1][1]-self.translation_bounds[1][0])

        for i in range(self.fluid.numActiveParticles()):
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

    def manip_state(self,state,tsp,spill_punish,target_fill):
        tsp_obs,spill_punish_obs,target_fill_obs = self._to_observations(tsp,spill_punish,target_fill)
        state[0][3],state[0][4],state[0][5] = tsp_obs,spill_punish_obs,target_fill_obs

if __name__=="__main__":
    observation_space = spaces.Tuple((spaces.Box(low=-1,high=1,shape=(1,)),spaces.Box(low=-1,high=1,shape=(400,))))
    print(observation_space[1].shape)