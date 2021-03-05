import gym
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import os,sys
import random
import argparse
from itertools import count

def human_player(env,slow=True,policy_uncertainty=0):
    """Try to pour in water yourself in the water-pouring environment.
    Use the arrow keys to move the bottle and W/S to rotate it.

    Args:
        env: The name of a water-pouring gym environment.
        slow: If true, the simulation won't run faster than real-time.
        policy_uncertainty: Amount of Signal dependent noise to add to performed actions.
    """
    env = gym.make(env,use_gui=True,policy_uncertainty=policy_uncertainty)
    if slow:
        step_time = env.time_step_size * env.steps_per_action
        start = time.perf_counter()
    tot_rew = 0
    t = 0
    for i in count(0):
        x,y,r = env.gui.get_bottle_x(),env.gui.get_bottle_y(),env.gui.get_bottle_rotation()
        observation,reward,done,info = env.step((r,x,y))
        rotation = R.from_matrix(env.bottle.rotation).as_euler("zyx")[0]
        translation_x,translation_y = env.bottle.translation[:2]
        tot_rew += reward
        t+=1
        env.render()
        if done:
            print("Reward",tot_rew,"\n Episode length",t)
            env.reset()
            tot_rew = 0
            t = 0
        if slow:
            left_time = i*step_time + start - time.perf_counter()
            if left_time>0:
                time.sleep(left_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="water_pouring:Pouring-mdp-v0", help="The OpenAI gym environment to use")
    parser.add_argument("--speed_mode", action="store_true", help="Run the Environment as fast as possible instead of in real time.")
    parser.add_argument("--policy_uncertainty", default=0, help="Amount of Signal dependent noise to add to performed actions.")
    args = parser.parse_args()
    human_player(args.env,not args.speed_mode, args.policy_uncertainty)