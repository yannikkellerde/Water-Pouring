import gym
from time import perf_counter
import numpy as np

def hard_mode():
    env = gym.make("water_pouring:Pouring-Base-v0",use_gui=True)

    for i in range(4000):
        x,y,r = env.gui.get_bottle_x(),env.gui.get_bottle_y(),env.gui.get_bottle_rotation()
        print(env.step((r/2+0.5,x/2+0.5,y/2+0.5,0)))
        env.render()

def simple_mode():
    env = gym.make("water_pouring:Pouring-Simple-v0",use_gui=True)

    for i in range(4000):
        r = env.gui.get_bottle_rotation()
        env.step((r/2+0.5,0))
        env.render()

if __name__ == "__main__":
    simple_mode()