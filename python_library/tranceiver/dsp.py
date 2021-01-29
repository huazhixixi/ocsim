from ..device_manager import device_selection
import numpy as np
import fractions


def _segment_axis(backend,a, length, overlap, mode='cut', append_to_end=0):
    """
        Generate a new array that chops the given array along the given axis into
        overlapping frames.
        example:
        >>> segment_axis(arange(10), 4, 2)
        array([[0, 1, 2, 3],
               [2, 3, 4, 5],
               [4, 5, 6, 7],
               [6, 7, 8, 9]])
        arguments:
        a       The array to segment must be 1d-array
        length  The length of each frame
        overlap The number of array elements by which the frames should overlap
        end     What to do with the last frame, if the array is not evenly
                divisible into pieces. Options are:
                'cut'   Simply discard the extra values
                'pad'   Pad with a constant value
        append_to_end:    The value to use for end='pad'
        a new array will be returned.
    """
    if a.ndim !=1:
        raise Exception("Error, input array must be 1d")
    if overlap > length:
        raise Exception("overlap cannot exceed the whole length")

    stride = length - overlap
    row = 1
    total_number = length
    while True:
        total_number = total_number + stride
        if total_number > len(a):
            break
        else:
            row = row + 1

    # 一共要分成row行
    if total_number > len(a):
        if mode == 'cut':
            b = backend.zeros((row, length), dtype=backend.complex128)
            is_append_to_end = False
        else:
            b = backend.zeros((row + 1, length), dtype=backend.complex128)
            is_append_to_end = True
    else:
        b = backend.zeros((row, length), dtype=backend.complex128)
        is_append_to_end = False

    index = 0
    for i in range(row):
        b[i, :] = a[index:index + length]
        index = index + stride

    if is_append_to_end:
        last = a[index:]

        b[row, 0:len(last)] = last
        b[row, len(last):] = append_to_end

    return b


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
    return rc


def rrcos_freq(backend,f, beta, T):
    return backend.sqrt(rcos_freq(backend,f, beta, T))

def pulseshaping(sig,beta):
    @device_selection(sig.device,True)
    def pulseshaping_(backend):
        f = backend.fft.fftfreq(sig.symbol_number * sig.sps_dsp) * sig.symbol_rate * sig.sps_dsp
        nyq_fil = rrcos_freq(backend, f, beta, 1 / sig.symbol_rate)
        nyq_fil /= nyq_fil.max()
        sig_f = backend.fft.fft(sig[:], axis=-1)
        sig_out = backend.fft.ifft(sig_f * nyq_fil, axis=-1)
        sig.samples = sig_out
        sig.roll_off = beta
        return sig
    return pulseshaping_()


def ideal_dac(signal,old_sps,new_sps):
    '''
        At current stage, only cpu is supported
    '''
    import resampy
    @device_selection(signal.device,True)
    def ideal_dac_(backend):
        assert signal.device == 'cpu'
        signal.samples = resampy.resample(signal.samples,old_sps,new_sps)
        return signal
    return ideal_dac_()




def cd_compensation(signal,span,fs):
    @device_selection(signal.device,True)
    def cdc_(backend):
        from scipy.constants import c
        center_wavelength = c / signal.center_freq
        freq_vector = backend.fft.fftfreq(len(signal[-1]), 1 /fs)
        omeg_vector = 2 * backend.pi * freq_vector
        if not isinstance(span,list):
            spans = [span]
        #
        else:
            spans = span
        for span_ in spans:
            beta2 = -span_.beta2(center_wavelength)
            dispersion = (-1j / 2) * beta2 * omeg_vector ** 2 * span_.length
            for row in signal[:]:
                row[:] = backend.fft.ifft(backend.fft.fft(row) * backend.exp(dispersion))

        return signal
    return cdc_()

