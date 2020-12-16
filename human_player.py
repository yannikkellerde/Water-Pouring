import gym
from time import perf_counter
import numpy as np
import os

def hard_mode():
    env = gym.make("water_pouring:Pouring-no-fix-v0",use_gui=True,uncertainty=0)

    for i in range(4000):
        x,y,r = env.gui.get_bottle_x(),env.gui.get_bottle_y(),env.gui.get_bottle_rotation()
        observation,reward,done,info = env.step((r,x,y))
        print(observation)
        env.render()
        if done:
            env.reset()

def simple_mode():
    env = gym.make("water_pouring:Pouring-simple-no-fix-v0",use_gui=True)
    
    full_rew = 0
    for i in range(4000):
        r = env.gui.get_bottle_rotation()
        obs,rew,done,info = env.step([r])
        full_rew += rew
        env.render()
        if done:
            print("\n\nYEEEEEEEEEEEEEEEEEEEEEEEEEEHAA\n",full_rew,"\n\n########################")
            env.reset()
            full_rew = 0

def test():
    env = gym.make("water_pouring:Pouring-no-fix-v0",use_gui=False)
    full_rew = 0
    done = False
    while not done:
        obs,rew,done,info = env.step([0])
        full_rew += rew
    print("FULL REW 1:",full_rew)
    #env.base.cleanup()
    #env = gym.make("water_pouring:Pouring-no-fix-v0",use_gui=True)
    env.reset(use_gui=True)
    full_rew = 0
    done = False
    while not done:
        obs,rew,done,info = env.step([0])
        env.render()
        full_rew += rew
    print("FULL REW 2:",full_rew)

if __name__ == "__main__":
    hard_mode()