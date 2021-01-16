from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
import numpy as np

class Model3d():
    def __init__(self,body,stretch_vertices=0):
        self.body = body
        self.orig_vertices = np.array(self.body.getGeometry().getVertices()).copy()
        self.hull_vertices = self.orig_vertices.copy()
        if stretch_vertices!=0: # Make more particles be counted as in glass
            vmax = np.max(self.hull_vertices[:,1])
            vmin = np.min(self.hull_vertices[:,1])
            diff = vmax-vmin
            to_stretch = stretch_vertices/diff
            self.hull_vertices[:,1] -= (vmin + vmax)/2
            self.hull_vertices[:,1] *= 1+to_stretch
            self.hull_vertices[:,1] += (vmin + vmax)/2
        self.hull = Delaunay(self.hull_vertices)
        self.orig_rect = np.array([[np.min(self.orig_vertices[:,i]),np.max(self.orig_vertices[:,i])] for i in range(3)])
        self.orig_rotation = np.array(self.body.getRotation()).copy()
        self.new_rotation = np.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]])
        self.rotation = self.orig_rotation@self.new_rotation
        self.orig_translation = np.array(self.body.getPosition()).copy()
        self.new_translation = np.array([0,0,0],dtype=np.float)
        self.translation = self.orig_translation + self.new_translation

    def rotate(self,rotation_matrix):
        self.new_rotation = self.new_rotation @ rotation_matrix
        self.rotation = self.orig_rotation@self.new_rotation
        self.body.setRotation(self.rotation)
        self.body.setWorldSpaceRotation(self.rotation)

    def translate(self,translation_vector):
        self.new_translation += translation_vector
        self.translation = self.orig_translation + self.new_translation
        self.body.setPosition(self.translation)
        self.body.setWorldSpacePosition(self.translation)

    def check_if_in_rect(self,other_model,to_translate,rot_radians):
        def check_points(points):
            bools = np.ones(len(points),dtype=np.bool)
            bools &= points[:,0] > self.orig_rect[0][0]
            bools &= points[:,0] < self.orig_rect[0][1]
            bools &= points[:,1] > self.orig_rect[1][0]
            bools &= points[:,1] < self.orig_rect[1][1]
            bools &= points[:,2] > self.orig_rect[2][0]
            bools &= points[:,2] < self.orig_rect[2][1]
            return bools
        to_rotate = R.from_euler("z",rot_radians).as_matrix()
        p = other_model.orig_vertices.copy()
        p -= other_model.orig_translation
        p = p@np.linalg.inv(other_model.new_rotation)@np.linalg.inv(to_rotate)
        p += other_model.orig_translation + other_model.new_translation + to_translate
        bools = check_points(p)
        if not bools.any():
            return to_translate,rot_radians
        problem_p = p[bools]
        test_p = problem_p - np.array((to_translate[0],0,0))
        if not check_points(test_p).any():
            return np.array((0,to_translate[1],0)),rot_radians
        test_p = problem_p - np.array((0,to_translate[1],0))
        if not check_points(test_p).any():
            return np.array((to_translate[0],0,0)),rot_radians
        test_p = problem_p - (other_model.orig_translation + other_model.new_translation + to_translate)
        test_p = test_p@to_rotate
        test_p += other_model.orig_translation + other_model.new_translation + to_translate
        if not check_points(test_p).any():
            return to_translate,0
        return np.array((0,0,0)),0

