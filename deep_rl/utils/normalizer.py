#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np
import torch
from ..utils.misc import RunningMeanStd
from ..utils.tiles_wrapper import TileCoder


class BaseNormalizer:
    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return


class MeanStdNormalizer(BaseNormalizer):
    def __init__(self, read_only=False, clip=10.0, epsilon=1e-8):
        BaseNormalizer.__init__(self, read_only)
        self.read_only = read_only
        self.rms = None
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x):
        x = np.asarray(x)
        if self.rms is None:
            self.rms = RunningMeanStd(shape=(1,) + x.shape[1:])
        if not self.read_only:
            self.rms.update(x)
        return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
                       -self.clip, self.clip)

    def state_dict(self):
        return {'mean': self.rms.mean,
                'var': self.rms.var}

    def load_state_dict(self, saved):
        self.rms.mean = saved['mean']
        self.rms.var = saved['var']


class RescaleNormalizer(BaseNormalizer):
    def __init__(self, config):
        BaseNormalizer.__init__(self)
        self.coef = config.coef

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        return self.coef * x


class ImageNormalizer(RescaleNormalizer):
    def __init__(self, config=None):
        config.coef = 1.0 / 255
        RescaleNormalizer.__init__(self, config)


class SignNormalizer(BaseNormalizer):
    def __init__(self, config=None):
        BaseNormalizer.__init__(self)
    
    def __call__(self, x):
        return np.sign(x)


class TileCodingNormalizer(BaseNormalizer):
    def __init__(self, config):
        BaseNormalizer.__init__(self)
        self.tiles_rep = TileCoder(config)
        self.dim = config.tiles_memsize

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        if x.shape.__len__() == 2:
            y = []
            for i in range(x.shape[0]):
                y.append(self.tiles_rep.get_representation(x[i]))
            y = np.asarray(y)
        else:
            y = self.tiles_rep.get_representation(x)
        y = 1 - y
        return y


class DummyNormalizer(BaseNormalizer):
    def __init__(self, config):
        BaseNormalizer.__init__(self)
        self.dim = config.observation_space.shape[0]

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        return x
    

class RandomFeatureNormalizer(BaseNormalizer):
    def __init__(self, config):
        BaseNormalizer.__init__(self)
        self.dim = config.random_feature_dim

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        y = []
        for i in range(x.shape[0]):
            np.random.seed(np.dot(np.geomspace(1, 1000, num=4).reshape(1, 4), x.reshape(4, 1)).astype(int).squeeze())
            y.append(np.random.choice(2, self.dim, p=[0.95, 0.05]))
        y = np.asarray(y)
        return y