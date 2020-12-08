import pysplishsplash
import pysplishsplash.Utilities.SceneLoaderStructs as Scene
from pysplishsplash.Extras import Scenes
import os
import numpy as np

def get_functions(obj):
    return [method_name for method_name in dir(obj)
                    if callable(getattr(obj, method_name))]


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
    print(base.getBoundarySimulator)
    base.initBoundaryData()
    print("HEEEYO",sim.getCurrent().numberOfBoundaryModels())
    bottle = sim.getCurrent().getBoundaryModel(2).getRigidBodyObject()
    print(bottle.getPosition(),bottle.getRotation(),bottle.getWorldSpacePosition())
    rotation_matrix = np.array([[0.9998477, -0.0174524,  0.0000000],
                                [0.0174524,  0.9998477,  0.0000000],
                                [0.0000000,  0.0000000,  1.0000000]])
    for i in range(1000):
        cur_rot = np.array(bottle.getRotation())
        print(cur_rot)
        new_rot = cur_rot@rotation_matrix
        bottle.setRotation(new_rot)
        base.timeStepNoGUI()
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