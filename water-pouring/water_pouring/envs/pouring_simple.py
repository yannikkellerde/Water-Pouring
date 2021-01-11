from pouring_base import Pouring_base
from gym import spaces
import math
import numpy as np
import os,sys

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
class Pouring_simple(Pouring_base):
    def __init__(self,**kwargs):
        super(Pouring_simple, self).__init__(scene_base=os.path.join(FILE_PATH,"scenes","simple_scene.json"),**kwargs)
        self.action_space = spaces.Box(low=-1,high=1,shape=(1,))
        self.observation_space = spaces.Box(low=0,high=1,shape=(4,))
    def step(self,action):
        return super(Pouring_simple,self).step((action[0],0,0))
    def _observe(self):
        rotation,translation_x,translation_y,in_bottle,fill_state,water_flow_estimate = super(Pouring_simple, self)._observe()
        return rotation,in_bottle,fill_state,water_flow_estimate