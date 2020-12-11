import gym
from time import perf_counter
env = gym.make("water_pouring:Pouring-Deterministic-v0",use_gui=True)

start = perf_counter()
for i in range(2000):
    x,y,r = env.gui.get_bottle_x(),env.gui.get_bottle_y(),env.gui.get_bottle_rotation()
    env.step((r/2+0.5,x/2+0.5,y/2+0.5,0))
    env.render()
print("EEEEEEEEEEEEEEeOOO",perf_counter()-start)