from typing import List
from ..core import WdmSignal
from ..device_manager import device_selection

import numpy as np
class Laser:

    def __init__(self,
                 linewidth: float
                 , fo: float
                 , power: float
                 , center_frequency: float):
        
        self.linewidth = linewidth
        self.fo = fo
        self.power = power

        backend = None
        self.phase_noise = None

        self.center_frequency = center_frequency

    def prop(self, signal):
        @device_selection(signal.device,True)
        def prop_(backend,signal,laser):
            var = 2 * backend.pi * laser.linewidth / signal.sps_in_fiber /signal.symbol_rate
            f = backend.random.normal(scale=np.sqrt(var), size=(signal[:].shape[0], len(signal[0])))
            laser.phase_noise = backend.cumsum(f, axis=1)

            signal[:] *= backend.exp(1j * laser.phase_noise)
            time_array = backend.arange(len(signal[0])) * 1 / signal.sps_in_fiber/signal.symbol_rate
            signal *= backend.exp(1j * 2 * backend.pi * laser.fo * time_array)
            laser.set_power(signal)
            return signal
        return prop_(signal,self)

    def set_power(self, signal):
        @device_selection(signal.device,True)
        def set_power_(backend,signal,laser):
            signal.normalize()

            signal [:]= backend.sqrt(10 ** (laser.power / 10) / 1000 / 2)
            signal.center_freq = laser.center_frequency
            return signal
        return set_power_(signal,self)

    def plot_phase_noise(self):
        try:
            phase_noise =np.array( self.phase_noise)
        except Exception:
            import cupy as cp
            phase_noise = cp.asnumpy(self.phase_noise)
        import matplotlib.pyplot as plt
        plt.subplots(121)
        plt.plot(phase_noise[0])
        plt.subplots(122)
        plt.plot(phase_noise[1])

class Multiplex:

    def __init__(self, wdm_center_freq):
        self.wdm_center_freq = wdm_center_freq
        self.device = None

    def mux(self, signals):
        devices = set()
        for signal in signals:
            devices.add(signal.device)

        if len(devices) != 1:
            raise Exception("All signals must be on the same devices")

        @device_selection(signal.device,True)
        def mux_(backend,multiplexer):

            freqs = np.array([signal.center_freq for signal in signals])
            band_freqs = freqs - multiplexer.wdm_center_freq

            t = backend.arange(len(signals[0][0])) * (1 / signals[0].symbol_rate/signals[0].sps_in_fiber)
            freqs = [signal.center_freq for signal in signals]
            cnt = 0
            samples = None
            for signal, freq in zip(signals, band_freqs):
                signal = signal * backend.exp(1j * 2 * backend.pi * freq * t)
                if not cnt:
                    samples = signal[:]
                    cnt += 1
                else:
                    samples += signal[:]

            symbols = [signal.symbol for signal in signals]

            #def __init__(self, symbols, samples, freqes, center_freq, sps_in_fiber, sps_dsp, device):

            wdm_signal = WdmSignal( symbols,samples, freqs, multiplexer.wdm_center_freq,  signal.sps_in_fiber,signal.sps_dsp
                                   ,signal.device)
            return wdm_signal
        return mux_(self)