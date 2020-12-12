from pouring_base import Pouring_base

class Pouring_uncertain(Pouring_base):
    def __init__(self,use_gui=False):
        super(Pouring_uncertain, self).__init__(use_gui=use_gui)
        self.rotation_uncertainty = 0.03
        self.translation_uncertainty = 0.03
        self.observation_uncertainty = 0.03