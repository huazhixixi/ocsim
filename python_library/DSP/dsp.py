import fractions

from ..device_manager import device_selection
def resamplingfactors(fold, fnew):
    ratn = fractions.Fraction(fnew / fold).limit_denominator()
    return ratn.numerator, ratn.denominator


def rcos_freq(backend,f, beta, T):
    """Frequency response of a raised cosine filter with a given roll-off factor and width """
    rc = backend.zeros(f.shape[0], dtype=f.dtype)
    rc[backend.where(backend.abs(f) <= (1 - beta) / (2 * T))] = T
    idx = backend.where((backend.abs(f) > (1 - beta) / (2 * T)) & (backend.abs(f) <= (
            1 + beta) / (2 * T)))
    rc[idx] = T / 2 * (1 + backend.cos(backend.pi * T / beta *
                                  (backend.abs(f[idx]) - (1 - beta) /
                                   (2 * T))))

    rc.shape = -1,1
    rc = backend.asfortranarray(rc)
    return rc


def rrcos_freq(backend,f, beta, T):
    return backend.sqrt(rcos_freq(backend,f, beta, T))



###################################################Pulse Shaping#######################################################
def pulse_shaping(beta,sig):

    @device_selection(sig.device)
    def pulse_shaping_():
        if 'cuda' in sig.device:
            import cupy as backend
        if 'cpu' in sig.device:
            import numpy as backend
        else:
            raise Exception('Error')

        f = backend.asfortranarray(backend.fft.fftfreq(sig.nsymb * sig.sps_dsp) * sig.symbol_rate * sig.sps_dsp)
        nyq_fil = rrcos_freq(backend, f, beta, 1 / sig.symbol_rate)
        nyq_fil /= nyq_fil.max()
        sig_f = backend.asfortranarray(backend.fft.fft(sig.samples, axis=0))
        sig_out = backend.asfortranarray(backend.fft.ifft(sig_f * nyq_fil, axis=0))
        sig.samples = sig_out
        return sig

    return pulse_shaping_()

###################################################Pulse Shaping#######################################################




###################################################Ideal resample#######################################################
def ideal_resample(signal):
    up, down = resamplingfactors(signal.sps_dsp,signal.sps_in_fiber)
    @device_selection(signal.device)
    def ideal_resample_():
        if 'cuda' in signal.device:
            import cusignal
            from cupy import asfortranarray
            signal.samples = cusignal.resample_poly(signal[:], up, down, axis=0)
            signal.samples = asfortranarray(signal.samples)

        if 'cpu' in signal.device:
            from scipy.signal import resample_poly
            from numpy import asfortranarray
            signal.samples = asfortranarray(resample_poly(signal[:], up, down, axis=0))

        return signal

    return ideal_resample_()
###################################################Ideal resample#######################################################




##################################################CD Compensation#######################################################





##################################################CD Compensation#######################################################
