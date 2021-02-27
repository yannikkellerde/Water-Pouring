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
                           "mapThickness":-0.01},
             "glas.obj":{"max_particles":275,
                         "mapThickness":-0.014},
             "beer.obj":{"max_particles":295,
                         "mapThickness":-0.01},
             "cocktail.obj":{"max_particles":140,
                            "mapThickness":-0.006}
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

def get_fluid_path(in_scene):
    with open(in_scene,"r") as f:
        scene = json.load(f)
    return scene["FluidModels"][0]["particleFile"]

def approx_3rd_deriv(f_x0,f_x0_minus_1h,f_x0_minus_2h,f_x0_minus_3h,h):
    return (1*f_x0-3*f_x0_minus_1h+3*f_x0_minus_2h-1*f_x0_minus_3h)/(h**3)

def approx_2rd_deriv(f_x0,f_x0_minus_1h,f_x0_minus_2h,h):
    return (-1*f_x0+2*f_x0_minus_1h-1*f_x0_minus_2h)/(h**2)

#def glass_to_glass(in_scene,out_file,env,glass):
