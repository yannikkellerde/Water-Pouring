from pouring_deterministic import Pouring_deterministic

class Pouring_nondeterministic(Pouring_deterministic):
    def __init__(self,gui_mode=False):
        super(Pouring_deterministic, self).__init__()
        self.rotation_uncertainty = 0.03
        self.translation_uncertainty = 0.03
        self.observation_uncertainty = 0.03