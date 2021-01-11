from pouring_simple import Pouring_simple
from gym import spaces
from scipy.spatial.transform import Rotation as R
import math
import numpy as np
import os,sys

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
class Pouring_mdp(Pouring_simple):
    def __init__(self,**kwargs):
        super(Pouring_mdp, self).__init__(**kwargs)
        self.action_space = spaces.Box(low=-1,high=1,shape=(1,))
        #self.observation_space = spaces.Tuple(spaces.Box(low=-1,high=1,shape=(1,)),
        #                                      spaces.Box(low=-1,high=1,shape=(4,self.max_particles)))
        self.observation_space = spaces.Tuple((spaces.Box(low=-1,high=1,shape=(1,)),
                                               spaces.Box(low=-1,high=1,shape=(self.max_particles,7))))
    def _observe(self):
        fluid_data = []
        rotation = R.from_matrix(self.bottle.rotation).as_euler("zyx")[0]
        rotation = (rotation-self.min_rotation)/(math.pi-self.min_rotation)
        for i in range(self.fluid.numActiveParticles()):
            pos = self.fluid.getPosition(i)
            vel = self.fluid.getVelocity(i)
            fluid_data.append((pos[0]/2,pos[1]/2,pos[2]/2,vel[0],vel[1],vel[2],rotation))
        fluid_data = np.clip(fluid_data,-1,1)
        feat_dat = np.array([rotation])
        return feat_dat,fluid_data

if __name__=="__main__":
    observation_space = spaces.Tuple((spaces.Box(low=-1,high=1,shape=(1,)),spaces.Box(low=-1,high=1,shape=(400,))))
    print(observation_space[1].shape)