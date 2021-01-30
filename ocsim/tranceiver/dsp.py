from ..device_manager import device_selection
from scipy.fft import fftfreq
from ..core import Signal


def rcos_freq(backend, f, beta, T):
    """Frequency response of a raised cosine filter with a given roll-off factor and width """
    rc = backend.zeros(f.shape[0], dtype=f.dtype)
    rc[backend.where(backend.abs(f) <= (1 - beta) / (2 * T))] = T
    idx = backend.where((backend.abs(f) > (1 - beta) / (2 * T)) & (backend.abs(f) <= (
            1 + beta) / (2 * T)))
    rc[idx] = T / 2 * (1 + backend.cos(backend.pi * T / beta *
                                       (backend.abs(f[idx]) - (1 - beta) /
                                        (2 * T))))
    return rc


def rrcos_freq(backend, f, beta, T):
    """Frequency transfer function of the square-root-raised cosine filter with a given roll-off factor and time width/sampling period after _[1]
    Parameters
    ----------
    f   : array_like
        frequency vector
    beta : float
        roll-off factor needs to be between 0 and 1 (0 corresponds to a sinc pulse, square spectrum)
    T   : float
        symbol period
    Returns
    -------
    y   : array_like
       filter response
    References
    ----------
    ..[1] B.P. Lathi, Z. Ding Modern Digital and Analog Communication Systems
    """
    return backend.sqrt(rcos_freq(backend, f, beta, T))


class PulseShaping:

    def __init__(self, beta):
        self.beta = beta

    def __core(self, signal: Signal) -> Signal:
        @device_selection(signal.device, True)
        def core_real(backend):
            f = backend.fft.fftfreq(signal.shape[1], 1 / signal.fs)
            h = rrcos_freq(backend, f, self.beta, 1 / signal.symbol_rate)
            h = h / h.max()
            signal[:] = backend.fft.ifft(backend.fft.fft(signal[:], axis=-1) * h)
            return signal

        return core_real()

    def __call__(self, signal) -> Signal:
        return self.__core(signal)


class IdealResampler:

    def __init__(self, old_sps, new_sps):
        self.old_sps = old_sps
        self.new_sps = new_sps

    def __core(self, signal: Signal) -> Signal:
        @device_selection(signal.device, True)
        def core_real(backend):
            if 'cuda' in signal.device:
                raise NotImplementedError
            import resampy
            signal.samples = resampy.resample(signal[:], self.old_sps, self.new_sps, filter="kaiser_best")
            signal.sps = self.new_sps
            return signal

        return core_real()

    def __call__(self, signal: Signal) -> Signal:
        return self.__core(signal)


class CDC:

    def __init__(self, fiber_setting):

        self.fiber_setting = fiber_setting

    def __core(self):
        pass

    def __call__(self, signal):
        pass





