from hw_tts.model.common.predictors import BasePredictor
from hw_tts.model.common.lr import LengthRegulator

class VarianceAdaptor(nn.Module):
    def __init__(self, *args, **kwargs):
        self.length_regulator = LengthRegulator(...)
        self.pitch_predictor = BasePredictor(...)
        self.energy_predictor = BasePredictor(...)
        self.duration_predictor = BasePredictor(...)


    def forward(self, x):
        pass
