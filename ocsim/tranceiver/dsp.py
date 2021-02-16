import fractions

from ..core import Signal
from ..device_manager import device_selection


def resamplingfactors(fold, fnew):
    ratn = fractions.Fraction(fnew / fold).limit_denominator()
    return ratn.numerator, ratn.denominator


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


def rrcos_time(backend, alpha, span, sps):
    '''
    Function:
        calculate the impulse response of the RRC
    Return:
        b,normalize the max value to 1
    '''
    assert divmod(span * sps, 2)[1] == 0
    M = span / 2
    n = backend.arange(-M * sps, M * sps + 1)
    b = backend.zeros(len(n))
    sps *= 1
    a = alpha
    Ns = sps
    for i in range(len(n)):
        if abs(1 - 16 * a ** 2 * (n[i] / Ns) ** 2) <= backend.finfo(backend.float).eps / 2:
            b[i] = 1 / 2. * ((1 + a) * backend.sin((1 + a) * backend.pi / (4. * a)) - (1 - a) * backend.cos(
                (1 - a) * backend.pi / (4. * a)) + (4 * a) / backend.pi * backend.sin((1 - a) * backend.pi / (4. * a)))
        else:
            b[i] = 4 * a / (backend.pi * (1 - 16 * a ** 2 * (n[i] / Ns) ** 2))
            b[i] = b[i] * (backend.cos((1 + a) * backend.pi * n[i] / Ns) + backend.sinc((1 - a) * n[i] / Ns) * (
                    1 - a) * backend.pi / (
                                   4. * a))
    return b / backend.sqrt(backend.sum(backend.abs(b) ** 2))


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
                import resampy
                signal.to('cpu')
                signal.samples = resampy.resample(signal[:], self.old_sps, self.new_sps, filter="kaiser_best")
                signal.sps = self.new_sps
                signal.to('cuda')
                return signal
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
        if not isinstance(fiber_setting, list):
            self.fiber_setting = [self.fiber_setting]

    def __core(self, signal):
        @device_selection(signal.device, True)
        def core_real(backend):
            from scipy.constants import c
            center_wavelength = c / signal.center_freq
            freq_vector = backend.fft.fftfreq(len(signal[0]), 1 / signal.fs)
            omeg_vector = 2 * backend.pi * freq_vector
            for span in self.fiber_setting:
                beta2 = -span.beta2(center_wavelength)
                dispersion = (-1j / 2) * beta2 * omeg_vector ** 2 * span.length
                for row in signal[:]:
                    row[:] = backend.fft.ifft(backend.fft.fft(row) * backend.exp(dispersion))
            return signal

        return core_real()

    def __call__(self, signal):
        return self.__core(signal)


class Equalizer:

    def __init__(self, tap_number):
        import numpy as np
        self.wxx = np.zeros((1, tap_number), dtype=np.complex)
        self.wxy = np.zeros((1, tap_number), dtype=np.complex)

        self.wyx = np.zeros((1, tap_number), dtype=np.complex)

        self.wyy = np.zeros((1, tap_number), dtype=np.complex)
        self.wxx[0, tap_number // 2] = 1
        self.wyy[0, tap_number // 2] = 1


class LmsPll(Equalizer):

    def __init__(self, tap_number, g, lr_train, total_loop, train_loop, lr_dd=None):
        self.tap_number = tap_number
        self.lr_train = lr_train
        self.lr_dd = lr_dd
        # if train_loop < total_loop:
        #     assert lr_dd is not None
        #     self.lr_dd = lr_dd
        self.total_loop = total_loop
        self.train_loop = train_loop
        self.g = g
        super(LmsPll, self).__init__(tap_number)

    def prop(self, signal):
        from .utilities import _segment_axis
        import numpy as np
        from .numba_backend import lms_equalize_core_pll
        train_symbol = np.asarray(signal.symbol[:, self.tap_number // 2 // signal.sps:], order='C')
        samples_xpol = _segment_axis(signal[0], self.tap_number, self.tap_number - signal.sps)
        samples_ypol = _segment_axis(signal[1], self.tap_number, self.tap_number - signal.sps)
        self.error_xpol_array = np.zeros((self.total_loop, len(samples_xpol)))
        self.error_ypol_array = np.zeros((self.total_loop, len(samples_xpol)))

        for idx in range(self.total_loop):
            is_train = idx < self.train_loop
            symbols, self.wxx, self.wxy, self.wyx, self.wyy, error_xpol_array, error_ypol_array \
                = lms_equalize_core_pll(samples_xpol, samples_ypol, self.g, train_symbol, self.wxx, self.wyy, self.wxy,
                                        self.wyx, self.lr_train, self.lr_dd, is_train)

            self.error_xpol_array[idx] = np.abs(error_xpol_array[0]) ** 2
            self.error_ypol_array[idx] = np.abs(error_ypol_array[0]) ** 2

        signal.symbol = train_symbol[:, :symbols.shape[1]]
        signal.samples = symbols
        return signal

    def __call__(self, signal):
        from ..utilities import cpu
        with cpu(signal) as signal:
            return self.prop(signal)


class CPE:
    pass
