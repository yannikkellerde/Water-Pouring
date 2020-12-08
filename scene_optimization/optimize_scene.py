from scene_creator import Scene_creator
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import numpy as np
import os
from copy import deepcopy
import subprocess
import shutil
import json
from partio_utils import evaluate_partio

class Optimizer():
    def __init__(self):
        self.scene_creator = Scene_creator()
        self.file_abs_path = os.path.abspath(os.path.dirname(__file__))
        self.simulator_path = "../../SPlisHSPlasH-2.8.7/bin/SPHSimulator"
        self.tmp_dir = os.path.join(self.file_abs_path,"tmp")
        self.output_dir = os.path.join(self.tmp_dir,"outputs")
        os.makedirs(self.output_dir,exist_ok=True)
        self.reward_map = {
            "spilled":-5,
            "glas":1,
            "bottle":0,
            "frames":-1
        }
        self.push_in_ranges = {
            "x":(-0.1,0.2),
            "y":(-0.05,0.2),
            "t1":(-3,4),
            "t2":(-3,4),
            "t3":(-4,4),
            "p1":(-0.15,0.15),
            "p2":(-0.15,0.15)
        }
        self.pbounds = {
            "x":(0,1),
            "y":(0,1),
            "t1":(0,1),
            "t2":(0,1),
            "t3":(0,1),
            "p1":(0,1),
            "p2":(0,1)
        }
        self.simulations_per_evaluation = 4
        if os.path.isfile("bayesian_optimization.log.json"):
            shutil.copyfile("bayesian_optimization.log.json","old_probes.json")
        self.optimizer = BayesianOptimization(
            f=self.eval_params,
            pbounds=self.pbounds,
            random_state=1,
        )
        logger = JSONLogger(path="bayesian_optimization.log")
        self.optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    def eval_scene(self,scene):
        sequence_times = np.array(scene["TargetAngleMotorHingeJoints"][0]["targetSequence"])[::2]
        for pre,now in zip(sequence_times,sequence_times[1:]):
            if pre >= now:
                return -50000

        scene_path = os.path.join(self.tmp_dir,"scene.json")
        with open(scene_path,"w") as f:
            json.dump(scene,f)
        shutil.rmtree(self.output_dir,ignore_errors=True)
        os.makedirs(self.output_dir,exist_ok=True)
        subprocess.run([
                self.simulator_path, '--no-gui',
                '--output-dir', self.output_dir, scene_path])
        stats = evaluate_partio(os.path.join(self.output_dir,"partio"))
        print(stats)
        score = sum([self.reward_map[key]*stats[key] for key in self.reward_map.keys()])
        return score

    def eval_params(self,**params_orig):
        params = deepcopy(params_orig)
        for key in params:
            params[key] = self.push_in_ranges[key][0]+params[key]*(self.push_in_ranges[key][1]-self.push_in_ranges[key][0])
        translation = [params["x"],params["y"]]
        movement = [params["t1"],params["p1"],params["t2"],params["p2"],params["t3"]]
        clean_scene = self.scene_creator.parametrize_scene(translation,movement)

        scores = []
        for _ in range(self.simulations_per_evaluation):
            scene = self.scene_creator.noisify_scene(clean_scene)
            scores.append(self.eval_scene(scene))
        return sum(scores)/len(scores)

    def load_from_json(self,fname):
        with open(fname,"r") as f:
            for line in f.read().splitlines():
                vals = json.loads(line)
                self.optimizer.register(params=vals["params"],target=vals["target"])

    def optimize(self,with_init_points=True):
        self.optimizer.maximize(
            init_points=10 if with_init_points else 0,
            n_iter=2000,
        )

if __name__ == "__main__":
    opt = Optimizer()
    opt.load_from_json("old_probes.json")
    opt.optimize(with_init_points=False)
    #print(opt.eval_params(p1=0.15, p2=0.15, t1=-0.44423454241730254, t2=1.4154392968409364, t3=0.36366214401167085, x=-0.1, y=0.2))