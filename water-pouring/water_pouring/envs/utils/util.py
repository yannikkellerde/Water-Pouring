import math
import numpy as np

def in_hull(p, hull, hull_orig_translation, hull_new_translation, hull_new_rotation):
    new_p = p-hull_orig_translation-hull_new_translation
    new_p = new_p @ hull_new_rotation
    new_p += hull_orig_translation
    return hull.find_simplex(new_p)>=0