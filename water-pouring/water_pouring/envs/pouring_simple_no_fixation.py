from pouring_base import Pouring_base
from gym import spaces
import math
import numpy as np
import os,sys

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
class Pouring_simple_no_fix(Pouring_base):
    def __init__(self,use_gui=False,uncertainty=0):
        super(Pouring_simple_no_fix, self).__init__(use_gui=use_gui,uncertainty=uncertainty,scene_file=os.path.join(FILE_PATH,"scenes","simple_scene.json"))
        self.action_space = spaces.Box(low=-1,high=1,shape=(1,))
        self.observation_space = spaces.Box(low=0,high=1,shape=(4,))
    def step(self,action):
        res = super(Pouring_simple_no_fix,self).step((action[0],0.5,0.5,1))
        self.fill_observation = self.particle_locations["glas"]
        return res
    def _observe(self):
        rotation,translation_x,translation_y,fixation,in_bottle,fill_state,water_flow_estimate = super(Pouring_simple_no_fix, self)._observe()
        return rotation,in_bottle,fill_state,water_flow_estimate