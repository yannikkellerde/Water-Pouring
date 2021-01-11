import math
import numpy as np
import json
import os

def in_hull(p, hull, hull_orig_translation, hull_new_translation, hull_new_rotation):
    new_p = p-hull_orig_translation-hull_new_translation
    new_p = new_p @ hull_new_rotation
    new_p += hull_orig_translation
    return hull.find_simplex(new_p)>=0

def manip_scene_file(in_scene,out_file,glas=None,bottle=None,fluid=None):
    with open(in_scene,"r") as f:
        scene = json.load(f)
    if glas is not None:
        scene["RigidBodies"][0]["geometryFile"] = os.path.join("../models/glasses/",glas)
    if bottle is not None:
        scene["RigidBodies"][1]["geometryFile"] = os.path.join("../models/bottles/",bottle)
    if fluid is not None:
        scene["FluidModels"][0]["particleFile"] = os.path.join("../models/fluids/",fluid)
    with open(out_file,"w") as f:
        json.dump(scene,f)