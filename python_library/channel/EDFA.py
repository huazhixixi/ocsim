import numpy as np
from ..device_manager import device_selection

class ConstantGainEdfa:

    def __init__(self,gain,nf):
        self.gain = gain
        self.nf = nf

    def prop(self,signal):
        @device_selection(signal.device,True)
        def prop_(backend,edfa):
            signal[:] = signal[:] * backend.sqrt(edfa.gain_linear)

            return signal

        return prop_(self)

    @property
    def gain_linear(self):
        return 10**(self.gain/10)

class ConstantPowerEdfa:

    def __init__(self,out_power,nf):
        self.nf = nf
        self.out_power = out_power

    def prop(self,signal):
        @device_selection(signal.device,True)
        def prop_(backed,edfa):

            power_now = backed.sum(backed.mean(backed.abs(signal[:]) ** 2, axis=-1))
            gain = edfa.out_power / power_now

            signal[:] = backed.sqrt(gain) * signal[:]

            return signal

        return prop_(self)
