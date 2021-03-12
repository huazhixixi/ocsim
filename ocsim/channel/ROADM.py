import numpy as np

from ..device_manager import device_selection


class WSS:

    def __init__(self, center_frequency, bandwidth, otf):

        self.center_freq = center_frequency
        self.bandwidth = bandwidth
        self.otf = otf

        self.H = None

    def get_transfer_function(self, freq_vector, device):

        @device_selection(device, True)
        def get_transfer_function_real(backend):

            if 'cuda' in device:
                from cupyx.scipy.special import erf
            else:
                from scipy.special import erf
            delta = self.otf / 2 / backend.sqrt(2 * backend.log(2))

            H = 0.5 * delta * backend.sqrt(2 * backend.pi) * (
                    erf((self.bandwidth / 2 - (freq_vector - self.center_freq)) / backend.sqrt(2) / delta) - erf(
                (-self.bandwidth / 2 - (freq_vector - self.center_freq)) / backend.sqrt(2) / delta))

            H = H / np.max(H)

            self.H = H

        get_transfer_function_real()

    def __core(self, signal):

        @device_selection(signal.device, True)
        def core_real(backend):
            freq = backend.fft.fftfreq(signal.shape[1], 1 / signal.fs)
            self.get_transfer_function(freq, signal.device)
            signal[:] = backend.fft.ifft(backend.fft.fft(signal[:], axis=-1) * self.H)
            return signal

        return core_real()

    def __call__(self, signal):

        return self.__core(signal)
