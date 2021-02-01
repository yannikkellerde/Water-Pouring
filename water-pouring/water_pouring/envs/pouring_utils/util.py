import math
import numpy as np
import json
import os

def in_hull(p, hull, hull_orig_translation, hull_new_translation, hull_new_rotation):
    new_p = p-hull_orig_translation-hull_new_translation
    new_p = new_p @ hull_new_rotation
    new_p += hull_orig_translation
    return hull.find_simplex(new_p)>=0


glass_map = {"normal.obj":{"max_particles":255,
                           "mapThickness":-0.05},
             "glas.obj":{"max_particles":275,
                         "mapThickness":-0.07},
             "beer.obj":{"max_particles":295,
                         "mapThickness":-0.05},
             "cocktail.obj":{"max_particles":140,
                            "mapThickness":-0.03}
}

def manip_scene_file(in_scene,out_file,env,glas=None,bottle=None,fluid=None):
    with open(in_scene,"r") as f:
        scene = json.load(f)
    if glas is not None:
        if glas in glass_map:
            env.max_in_glas = glass_map[glas]["max_particles"]
            scene["RigidBodies"][0]["mapThickness"] = glass_map[glas]["mapThickness"]
        scene["RigidBodies"][0]["geometryFile"] = os.path.join("../models/glasses/",glas)
    if bottle is not None:
        scene["RigidBodies"][1]["geometryFile"] = os.path.join("../models/bottles/",bottle)
    if fluid is not None:
        scene["FluidModels"][0]["particleFile"] = os.path.join("../models/fluids/",fluid)
    with open(out_file,"w") as f:
        json.dump(scene,f)

#def glass_to_glass(in_scene,out_file,env,glass):
