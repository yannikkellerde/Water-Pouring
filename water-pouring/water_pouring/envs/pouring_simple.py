from pouring_base import Pouring_base
from gym import spaces
import math
import numpy as np
import os,sys

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
class Pouring_simple(Pouring_base):
    def __init__(self,use_gui=False,uncertainty=0):
        super(Pouring_simple, self).__init__(use_gui=use_gui,uncertainty=uncertainty,scene_file=os.path.join(FILE_PATH,"scenes","simple_scene.json"))
        self.action_space = spaces.Box(low=-1,high=1,shape=(2,))
        self.observation_space = spaces.Box(low=0,high=1,shape=(5,))
    def step(self,action):
        return super(Pouring_simple,self).step((action[0],0.5,0.5,action[1]))
    def _observe(self):
        rotation,translation_x,translation_y,fixation,in_bottle,fill_state,water_flow_estimate = super(Pouring_simple, self)._observe()
        return rotation,fixation,in_bottle,fill_state,water_flow_estimate