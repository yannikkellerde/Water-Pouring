import gym
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import os,sys

def hard_mode():
    env = gym.make("water_pouring:Pouring-mdp-full-v0",use_gui=True,policy_uncertainty=0.3)
    step_time = env.time_step_size * env.steps_per_action
    start = time.perf_counter()
    tot_rew = 0
    t = 0
    for i in range(4000):
        x,y,r = env.gui.get_bottle_x(),env.gui.get_bottle_y(),env.gui.get_bottle_rotation()
        observation,reward,done,info = env.step((r,x,y))
        rotation = R.from_matrix(env.bottle.rotation).as_euler("zyx")[0]
        translation_x,translation_y = env.bottle.translation[:2]
        #print(translation_x,translation_y,rotation)
        #print(observation[0],np.min(observation[1],axis=0),np.max(observation[1],axis=0))
        print(env.particle_locations)
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

def with_evals():
    env = gym.make("water_pouring:Pouring-mdp-full-v0",use_gui=True,policy_uncertainty=0.3,fixed_spill=True,fixed_tsp=True,fixed_target_fill=True)
    step_time = env.time_step_size * env.steps_per_action
    start = time.perf_counter()
    td3_path = os.path.join(os.path.dirname(__file__),"../TD3")
    sys.path.append(td3_path)
    from q_eval_interface import q_eval_interface
    q_evaluator = q_eval_interface(env=env,model_path=sys.argv[1],norm="layer")
    tot_rew = 0
    t = 0
    fig = plt.figure(figsize=(3,6))
    plt.tight_layout()
    rects = plt.bar([0,1],[0,0])
    plt.ylabel("score")
    ax = plt.gca()
    plt.xticks([0,1],["Q value","total reward"])
    observation,done = env.reset(),False
    for i in range(4000):
        x,y,r = env.gui.get_bottle_x(),env.gui.get_bottle_y(),env.gui.get_bottle_rotation()
        action = (r,x,y)
        q = q_evaluator.eval_q(observation,action)
        observation,reward,done,info = env.step(action)
        rects[0].set_height(q)
        rects[1].set_height(tot_rew)
        bounds = [-200,200]
        rects[0].set_color(np.clip(((-q-bounds[0])/(bounds[1]-bounds[0]),(q-bounds[0])/(bounds[1]-bounds[0]),0),0,1))
        rects[1].set_color(np.clip(((-tot_rew-bounds[0])/(bounds[1]-bounds[0]),(tot_rew-bounds[0])/(bounds[1]-bounds[0]),0),0,1))
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.001)
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
    #with_evals()
    #hard_mode()
    g2g()
    #featured()
    #simple_mode()
    #mdp_mode()