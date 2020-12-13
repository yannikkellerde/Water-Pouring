import gym
from time import perf_counter
import numpy as np
import os
import psutil

def hard_mode():
    env = gym.make("water_pouring:Pouring-Base-v0",use_gui=True,uncertainty=0.02)

    for i in range(4000):
        x,y,r = env.gui.get_bottle_x(),env.gui.get_bottle_y(),env.gui.get_bottle_rotation()
        observation,reward,done,info = env.step((r/2+0.5,x/2+0.5,y/2+0.5,1))
        print(info)
        env.render()
        if done:
            env.reset()

def simple_mode():
    env = gym.make("water_pouring:Pouring-Simple-v0",use_gui=True)

    for i in range(4000):
        r = env.gui.get_bottle_rotation()
        print(env.step((r/2+0.5,1)))
        env.render()

def check_if_leaking_memory():
    def check_memory():
        process = psutil.Process(os.getpid())
        return process.memory_percent()

    env = gym.make("water_pouring:Pouring-Simple-v0",use_gui=False)
    wuff = 0
    while 1:
        action = env.action_space.sample()
        obs,rew,done,_ = env.step(action)
        if done:
            env.reset()
            wuff += 1
            if wuff%10==1:
                print(check_memory())




if __name__ == "__main__":
    check_if_leaking_memory()