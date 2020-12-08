from copy import deepcopy
import json
import os
import numpy as np

FILE_PATH = os.path.dirname(__file__)
class Scene_creator():
    def __init__(self):
        with open(os.path.join(FILE_PATH,"base_scene.json")) as f:
            self.base_scene = json.load(f)
    def noisify_scene(self,scene,level=1):
        scene=deepcopy(scene)
        scene["Configuration"]["gravitation"][2] = np.random.normal(scene["Configuration"]["gravitation"][2],level*0.5)

        bottle_translation = [np.random.normal(x,0.03*level) for x in scene["FluidModels"][0]["translation"]]
        scene["FluidModels"][0]["translation"] = bottle_translation

        for body in scene["RigidBodies"]:
            if body["id"]==1:
                body["scale"] = [np.random.normal(x,0.03*level) for x in body["scale"]]
                body["translation"][0] = np.random.normal(body["translation"][0],0.03*level)
                body["translation"][2] = np.random.normal(body["translation"][1],0.03*level)
            if body["id"] == 2:
                body["translation"] = bottle_translation
        joint = scene["TargetAngleMotorHingeJoints"][0]
        joint["position"] = [np.random.normal(x,0.01*level) for x in joint["position"]]
        for i,val in enumerate(joint["targetSequence"]):
            if i>=2 and i%2==0:
                joint["targetSequence"][i] = np.random.normal(val,0.1*level)
            elif i>=2 and i%2==1:
                joint["targetSequence"][i] = np.random.normal(val,0.01*level)
        return scene

    def parametrize_scene(self,translation,movement):
        scene = deepcopy(self.base_scene)
        joint = scene["TargetAngleMotorHingeJoints"][0]
        joint["position"][0] = joint["position"][0]+translation[0]
        joint["position"][1] = joint["position"][1]+translation[1]
        bottle = scene["RigidBodies"][2]
        bottle["translation"][0] = bottle["translation"][0]+translation[0]
        bottle["translation"][1] = bottle["translation"][1]+translation[1]
        water = scene["FluidModels"][0]
        water["translation"][0] = water["translation"][0]+translation[0]
        water["translation"][1] = water["translation"][1]+translation[1]

        joint["targetSequence"] = joint["targetSequence"][:2]+list(np.array(joint["targetSequence"][2:7])+np.array(movement))+joint["targetSequence"][-1:]
        scene["Configuration"]["stopAt"] = joint["targetSequence"][6]
        return scene