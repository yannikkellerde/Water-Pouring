import pysplishsplash
import pysplishsplash.Utilities.SceneLoaderStructs as Scene
from pysplishsplash.Extras import Scenes
from partio_utils import partio_write_rigid_body,partio_uncompress
import os,sys,time
import numpy as np
from scipy.spatial import Delaunay
from functools import reduce
from tqdm import tqdm,trange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_functions(obj):
    return [method_name for method_name in dir(obj)
                    if callable(getattr(obj, method_name))]


def set_axes_equal(ax):
  # Source https://stackoverflow.com/a/31364297

    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def main():
    base = pysplishsplash.Exec.SimulatorBase()
    sim = pysplishsplash.Simulation()
    args = base.init(useGui=False,outputDir=os.path.abspath("test"),sceneFile=os.path.abspath("base_scene.json"))
    #args = base.init(useGui=True,outputDir=os.path.abspath("test"),stateFile=os.path.abspath("state/state.bin"),sceneFile=Scenes.MotorScene)
    #gui = pysplishsplash.GUI.Simulator_GUI_imgui(base)
    #base.setGui(gui)
    #scene = base.getScene()
    #add_block = Scene.FluidBlock('Fluid', Scene.Box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), 0, [0.0, 0.0, 0.0])
    #scene.fluidBlocks[1] = add_block # In Place construction not supported yet
    base.setValueBool(base.PARTIO_EXPORT, True)
    base.setValueBool(base.RB_EXPORT,True)
    base.setValueBool(base.VTK_EXPORT, True)
    base.setValueBool(base.RB_VTK_EXPORT, True)
    base.setValueFloat(base.STOP_AT, 1.0)
    base.initSimulation()
    base.initBoundaryData()
    #print("HEEEYO",sim.getCurrent().numberOfBoundaryModels())
    bottle = sim.getCurrent().getBoundaryModel(2).getRigidBodyObject()
    #print(bottle.getPosition(),bottle.getRotation(),bottle.getWorldSpacePosition(),bottle.getWorldSpaceRotation(),sep="\n")
    rotation_matrix = np.array([[0.9999999, -0.0005236,  0.0000000],
                                [0.0005236,  0.9999999,  0.0000000],
                                [0.0000000,  0.0000000,  1.0000000]])
    #print("HEEEEELLOOOO",geo.numVertices())
    rigid_body_name = "test/partio_rigid/rigid_{}.bgeo"
    geo = bottle.getGeometry()
    verts = np.array(geo.getVertices())
    partio_write_rigid_body(verts,rigid_body_name.format(1))
    cur_rot = rotation_matrix
    bottle_rot_orig = np.array(bottle.getRotation())
    bottle_translation = np.array(bottle.getPosition())
    for i in trange(2000):
        new_rot = bottle_rot_orig@cur_rot
        bottle.setRotation(new_rot)
        bottle.setWorldSpaceRotation(new_rot)
        #geo.rotateVertices(rotation_matrix)
        #geo.updateNormals()
        #geo.updateVertexNormals()
        if i%8==0:
            verts_now = ((verts-bottle_translation) @ cur_rot.T)+bottle_translation
            partio_write_rigid_body(verts_now,rigid_body_name.format(int(i/8+2)))
        base.timeStepNoGUI()
        cur_rot = cur_rot@rotation_matrix

    partio_uncompress("test/partio")

    #print(bottle.getPosition(),bottle.getRotation(),bottle.getWorldSpacePosition(),bottle.getWorldSpaceRotation(),sep="\n")
    #base.saveState("state")
    #base.rigidBodyExport()
    #print(sim.getCurrent().numberOfBoundaryModels())
    #print(sim.getCurrent().numberOfFluidModels())
    #print(get_functions(sim.getCurrent().getFluidModel(0)))
    #print(sim.getCurrent().getBoundaryHandlingMethod())
    #print(base.getBoundarySimulator())
    #print(base.readBoundaryState())
    #base.particleExport()
    #base.run()
    #base.saveState("state")
    #base.writeRigidBodiesBIN("rigid_test")

if __name__ == "__main__":
    main()