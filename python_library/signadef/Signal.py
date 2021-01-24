import numpy as np
# from .. import CUDA_AVA
import contextlib

import types
from functools import wraps
from .utilities import normalize
from .utilities import to



class Signal(object):

    def __init__(self,
                 sps_in_fiber,
                 symbol_rate,
                 center_frequency,

                 samples,
                 device='cpu',
                 sps_dsp=None,

                 ):

        self.sps_in_fiber = sps_in_fiber
        self.symbol_rate = symbol_rate
        self.center_freq = center_frequency
        self.samples = samples
        self.sps_dsp = sps_dsp
        self.device = device
        self.engine = None
        self.to(self.device)

    def ensure_sample_dim(self):
        self.engine.atleast_2d(self.samples)

    @property
    def device_number(self):
        try:
            return int(self.device.split(':')[-1])
        except Exception:
            return None

    def to(self, device):
        return to(self, self.device)

    def __getitem__(self, item):
        return self.samples[item]

    def __setitem__(self, key, value):
        self.samples[key] = value

    def normalize(self):
        return normalize(self, self.device)

    @classmethod
    def load_from_mat(cls, name):
        pass


class QamSignal(Signal):

    def __init__(self,
                 qam_order,
                 symbol_number,
                 sps_in_fiber,
                 symbol_rate,
                 center_frequency,
                 device='cpu',
                 sps_dsp=None,
                 is_int=True,
                 pol_dim=2
                 ):
        self.qam_order = qam_order
        self.symbol_number = symbol_number
        self.sps_dsp = 2 if sps_dsp is None else sps_dsp
        self.pol_number = pol_dim

        if is_int:
            self.map()
            samples = np.zeros((self.symbol_number * self.sps_dsp, self.pol_number), order='F')
            super(QamSignal, self).__init__(sps_in_fiber=sps_in_fiber, symbol_rate=symbol_rate,
                                            center_frequency=center_frequency, device=device, samples=samples,
                                            sps_dsp=sps_dsp)

    def map(self):
        pass

    def scatterplot(self, sps=1):
        pass


if __name__ == '__main__':
    pass
