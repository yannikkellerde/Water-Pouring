from scipy.spatial import Delaunay
import numpy as np

class Model3d():
    def __init__(self,body):
        self.body = body
        self.orig_vertices = np.array(self.body.getGeometry().getVertices()).copy()
        self.hull = Delaunay(self.orig_vertices)
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