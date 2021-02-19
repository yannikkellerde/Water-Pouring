import gym
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

def hard_mode():
    env = gym.make("water_pouring:Pouring-mdp-full-v0",use_gui=True,policy_uncertainty=0.3)
    step_time = env.time_step_size * env.steps_per_action *3
    start = time.perf_counter()
    tot_rew = 0
    t = 0
    for i in range(4000):
        x,y,r = env.gui.get_bottle_x(),env.gui.get_bottle_y(),env.gui.get_bottle_rotation()
        observation,reward,done,info = env.step((r,x,y))
        rotation = R.from_matrix(env.bottle.rotation).as_euler("zyx")[0]
        translation_x,translation_y = env.bottle.translation[:2]
        #print(translation_x,translation_y,rotation)
        print(observation[0],np.min(observation[1],axis=0),np.max(observation[1],axis=0))
        #print(env.particle_locations)
        tot_rew += reward
        t+=1
        env.render()
        if done:
            print("Reward",tot_rew,"\n tsp",env.time_step_punish,"\n spill",env.spill_punish,"\n Time",t)
            env.reset()
            tot_rew = 0
            t = 0
        left_time = i*step_time + start - time.perf_counter()
        if left_time>0:
            time.sleep(left_time)
            #input()

def g2g():
    env = gym.make("water_pouring:Pouring-g2g-mdp-v0",use_gui=True,policy_uncertainty=0.3)
    step_time = env.time_step_size * env.steps_per_action
    start = time.perf_counter()
    tot_rew = 0
    t = 0
    for i in range(4000):
        x,y,r = env.gui.get_bottle_x(),env.gui.get_bottle_y(),env.gui.get_bottle_rotation()
        observation,reward,done,info = env.step((r,x,y))
        #print(observation)
        #print(env.particle_locations["air"])
        print(env.particle_locations)
        tot_rew += reward
        t+=1
        env.render()
        if done:
            print("Reward",tot_rew,"\n tsp",env.time_step_punish,"\n spill",env.spill_punish,"\n Time",t)
            #exit()
            env.reset()
            tot_rew = 0
            t = 0
        left_time = i*step_time+start - time.perf_counter()
        if left_time>0:
            time.sleep(left_time)

def featured():
    env = gym.make("water_pouring:Pouring-featured-v0",use_gui=True,policy_uncertainty=0.3)
    step_time = env.time_step_size * env.steps_per_action
    start = time.perf_counter()
    tot_rew = 0
    t = 0
    time.sleep(3)
    for i in range(4000):
        x,y,r = env.gui.get_bottle_x(),env.gui.get_bottle_y(),env.gui.get_bottle_rotation()
        observation,reward,done,info = env.step((r,x,y))
        print(observation)
        print(env.particle_locations["air"])
        tot_rew += reward
        t+=1
        env.render()
        if done:
            print("Reward",tot_rew,"\n tsp",env.time_step_punish,"\n spill",env.spill_punish,"\n Time",t)
            exit()
            env.reset()
            tot_rew = 0
            t = 0
        left_time = i*step_time+start - time.perf_counter()
        if left_time>0:
            time.sleep(left_time)

def simple_mode():
    env = gym.make("water_pouring:Pouring-simple-no-fix-v0",use_gui=True)
    step_time = env.time_step_size * env.steps_per_action
    start = time.perf_counter()
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
        left_time = i*step_time+start - time.perf_counter()
        if left_time>0:
            time.sleep(left_time)

def mdp_mode():
    env = gym.make("water_pouring:Pouring-mdp-v0",use_gui=True)
    
    full_rew = 0
    step_time = env.time_step_size * env.steps_per_action
    start = time.perf_counter()
    for i in range(4000):
        r = env.gui.get_bottle_rotation()
        obs,rew,done,info = env.step([r])
        full_rew += rew
        env.render()
        if done:
            print(full_rew)
            env.reset()
            full_rew = 0
        left_time = i*step_time+start - time.perf_counter()
        if left_time>0:
            time.sleep(left_time)

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
    #g2g()
    #featured()
    #simple_mode()
    #mdp_mode()