import numpy as np
from .. import CUDA_AVA


class Signal(object):

    def __init__(self,
                 sps_in_fiber,
                 symbol_rate,
                 center_frequency,

                 samples,
                 device = 'cpu',
                 sps_dsp = None,

                 ):

        self.sps_in_fiber = sps_in_fiber
        self.symbol_rate = symbol_rate
        self.center_freq = center_frequency
        self.samples = samples
        self.sps_dsp = sps_dsp
        self.device = device
        self.engine = None
        self.to(self.device)


    def device_context_manager(self):

        pass

    def ensure_sample_dim(self):
        self.engine.atleast_2d(self.samples)

    @property
    def device_number(self):
        return int(self.device.split(':')[-1])

    def to(self,device):

        if device == self.device:
            return self

        if 'cuda' in self.device:
            assert CUDA_AVA
            import cupy as cp
            with cp.cuda.Device(self.device_number):
                self.samples = cp.asarray(self.samples,order='F')
                self.device = device
                self.engine = cp
            return self

        if 'cpu' == self.device:
            import cupy as cp
            with cp.cuda.Device(self.device_number):
                self.samples = cp.asnumpy(self.samples,order='F')
                self.device = 'cpu'
                self.engine = np
            return self

    def __getitem__(self, item):
        return self.samples[item]

    def __setitem__(self, key, value):
        self.samples[key] = value

    def normalise(self):
        pass

