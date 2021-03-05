from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
import numpy as np

class Model3d():
    """A 3-dimensional object described by it's vertices that takes the role of
    an api to a SPlisHSPlasH representation of the object.

    Attributes:
        body: Reference to the SPlisHSPlasH RigidBody object this object is associated with.
        orig_vertices: numpy array of dimensions num_vertices X 3 that stores the positon of
                       the vertices at object creation time.
        hull_vertices: numpy array of dimensions num_vertices X 3 that stores the current positon
                       of the objects vertices
        hull: A scipy.spatial Delaunay hull for the object.
        orig_rect: The rectangle spanned by orig_vertices at creation time
        orig_rotation: The objects SPlisHSPlasH rotation at creation time in rotation matrix format.
        new_rotation: The amount the object has been rotated after creation in matrix format.
        rotation: The total current rotation of the object.
        orig_translation: A 3d Vector containing the objects SPlisHSPlasH translation at creation time.
        new_translation: A 3d Vector containing the amount the object has been translated after creation.
        translation: A 3d Vector containing the total current translation of the object.
    """
    def __init__(self,body,stretch_vertices=0):
        """Initialize Model3d api that communicates with SPlisHSPlasH

        Args:
            body: Reference to the SPlisHSPlasH RigidBody object this object will be associated with.
            stretch_vertices: Stretch out the vertices of the object in y direction by this amount.
        """
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
        """Rotates this object and the associated SPlisHSPlasH RigidBody object.

        Args:
            rotation_matrix: 3 x 3 numpy array describing the desired rotation.
        """
        self.new_rotation = self.new_rotation @ rotation_matrix
        self.rotation = self.orig_rotation@self.new_rotation
        self.body.setRotation(self.rotation)
        self.body.setWorldSpaceRotation(self.rotation)

    def translate(self,translation_vector):
        """Translate this object and the associated SPlisHSPlasH RigidBody object.

        Args:
            translation_vector: 3d translation vector describing the desired translation in x,y,z direction.
        """
        self.new_translation += translation_vector
        self.translation = self.orig_translation + self.new_translation
        self.body.setPosition(self.translation)
        self.body.setWorldSpacePosition(self.translation)

    def check_if_in_rect(self,other_model,to_translate,rot_radians):
        """Check if the another Model3d would collide with this object if it
        was translated and rotated by a specific amount. This method currently
        won't work correctly if this object has non zero new_rotation or new_translation.

        Args:
            other_model: A Model3d object.
            to_translation: 3d translation vector to consider as translation for other_model.
            rot_radians: 3 x 3 numpy array that describes a rotation to consider for other_model.
        
        Returns:
            The inputted to_translate and rot_radians parameter values, but translations or rotations
            that would result in a collision with this object are set to zero.
        """
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

