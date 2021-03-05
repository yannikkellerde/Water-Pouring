import math
import numpy as np
import json
import os
import random

def in_hull(p, hull, hull_orig_translation, hull_new_translation, hull_new_rotation):
    """Check if a point is inside a hull of an object

    Args:
        p: A 3d numpy vector describing the position of a point.
        hull: A scipy.spatial.Delauny instance.
        hull_orig_translation: The objects original translation (see Model3d class)
        hull_new_translation: The objects new translation (see Model3d class)
        hull_new_rotation: The objects new rotation (see Model3d class)

    Returns:
        True if the point is inside the hull of the object, False otherwise.
    """
    new_p = p-hull_orig_translation-hull_new_translation
    new_p = new_p @ hull_new_rotation
    new_p += hull_orig_translation
    return hull.find_simplex(new_p)>=0

def get_fluid_path(in_scene):
    """Extract the path to the fluid model that is used in a SPlisHSPlasH scene file.

    Args:
        in_scene: Path to a SPlisHSPlasH scene file
    Returns:
        The path to the fluid model file.
    """
    with open(in_scene,"r") as f:
        scene = json.load(f)
    return scene["FluidModels"][0]["particleFile"]

def approx_3rd_deriv(f_x0,f_x0_minus_1h,f_x0_minus_2h,f_x0_minus_3h,h):
    """Backwards numerical approximation of the third derivative of a function.

    Args:
        f_x0: Function evaluation at current timestep.
        f_x0_minus_1h: Previous function evaluation.
        f_x0_minus_2h: Function evaluations two timesteps ago.
        f_x0_minus_3h: Function evaluations three timesteps ago.
        h: Time inbetween function evaluations.

    Returns:
        The approximated value of the third derivative of f evaluated at x0
    """
    return (1*f_x0-3*f_x0_minus_1h+3*f_x0_minus_2h-1*f_x0_minus_3h)/(h**3)

def approx_2rd_deriv(f_x0,f_x0_minus_1h,f_x0_minus_2h,h):
    """Backwards numerical approximation of the second derivative of a function.

    Args:
        f_x0: Function evaluation at current timestep.
        f_x0_minus_1h: Previous function evaluation.
        f_x0_minus_2h: Function evaluations two timesteps ago.
        h: Time inbetween function evaluations.

    Returns:
        The approximated value of the second derivative of f evaluated at x0
    """
    return (-1*f_x0+2*f_x0_minus_1h-1*f_x0_minus_2h)/(h**2)

def extract_time_step_size(scene_file):
    """Extract the time_step_size from a SPlisHSPlasH scene file.

    Args:
        scene_file: Path to a SPlisHSPlasH scene file
    Returns:
        The time_step_size that is used in the scene file.
    """
    with open(scene_file,"r") as f:
        data = json.load(f)
    return float(data["Configuration"]["timeStepSize"])

def get_random(self,attr_range):
    """Get a random value in the specified range.

    Args:
        attr_range: A 2d List specifying the beginning and the end of the range.
    Returns:
        A random float between attr_range[0] and attr_range[1]
    """
    return random.random() * (attr_range[1]-attr_range[0]) + attr_range[0]