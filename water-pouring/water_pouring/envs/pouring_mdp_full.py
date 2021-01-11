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
        self.observation_space = spaces.Tuple((spaces.Box(low=-1,high=1,shape=(3,)),
                                               spaces.Box(low=-1,high=1,shape=(self.max_particles,9))))

        self.max_rotation_radians = 0.003 # Hyperparameter manipulation. Might want to remove

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
            fluid_data.append((pos[0]/2,pos[1]/2,pos[2]/2,vel[0],vel[1],vel[2],rotation,translation_x,translation_y))
        fluid_data = np.clip(fluid_data,-1,1)
        feat_dat = np.array([rotation,translation_x,translation_y])
        return feat_dat,fluid_data

if __name__=="__main__":
    observation_space = spaces.Tuple((spaces.Box(low=-1,high=1,shape=(1,)),spaces.Box(low=-1,high=1,shape=(400,))))
    print(observation_space[1].shape)