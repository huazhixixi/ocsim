import numpy as np
from ..device_manager import device_selection
from .utilities import  rescale_signal

def quantize_signal(signal, nbits=6, rescale_in=True, rescale_out=True):
    """
    Function so simulate limited resultion using DACs and ADCs, limit quantization error to (-delta/2,delta/2) and set
        decision threshold at mid-point between two quantization levels.
    Parameters:
        sig_in:            Input signal, numpy array, notice: input signal should be rescale to (-1,1)
        nbits:          Quantization resolution
        rescale_in:        Rescale input signal to (-1,1)
        rescale_out:       Rescale output signal to (-input_max_swing,input_max_swing)
    Returns:
        sig_out:        Output quantized waveform
    """
    # 2**nbits interval within (-1,1), output swing is (-1+delta/2,1-delta/2)
    # Create a 2D signal
    @device_selection(signal.device,True)
    def quantize_signal_real(backend):
        import copy
        sig_in = signal
        assert sig_in.samples.ndim ==2
        # sig_in = backend.atleast_2d(sig_in)
        npols = sig_in.shape[0]

        # Rescale to
        sig = backend.zeros((npols,sig_in.shape[1]), dtype=sig_in.dtype)
        if rescale_in:
            for pol in range(npols):
                sig[pol] = rescale_signal(sig_in[pol], sig_in.device,swing=1)

        swing = 2
        delta = swing/2**nbits
        levels_out = backend.linspace(-1+delta/2, 1-delta/2, 2**nbits)
        levels_dec = levels_out + delta/2

        sig_out = backend.zeros(sig.shape, dtype="complex")
        for pol in range(npols):
            sig_quant_re = levels_out[backend.digitize(sig[pol].real, levels_dec[:-1], right=False)]
            sig_quant_im = levels_out[backend.digitize(sig[pol].imag, levels_dec[:-1], right=False)]
            sig_out[pol] = sig_quant_re + 1j * sig_quant_im

        if not backend.iscomplexobj(sig):
            sig_out = sig_out.real

        if rescale_out:
            max_swing = backend.maximum(backend.abs(sig_in.real).max(), backend.abs(sig_in.imag).max())
            sig_out = sig_out * max_swing
        signal.samples = sig_out
        return signal
    return quantize_signal_real()

def mux(signals, center_freq=None):
    freq = [signal.center_freq for signal in signals]
    # device = [signal.device for signal in signals]
    length = np.diff([signal.shape[1] for signal in signals])
    fs = np.diff([signal.fs for signal in signals])

    # assert all(device)
    assert np.all(length == 0)
    assert np.all(fs == 0)

    if center_freq is None:
        center_freq = (np.min(freq) + np.max(freq)) / 2

    relative_freq = np.array(freq) - center_freq

    length = signals[0].shape[1]
    fs = signals[0].fs

    symbols = [signal.symbol for signal in signals]

    @device_selection(device[0], True)
    def mux_real(backend):
        temp = 0

        t = backend.arange(0, length) * 1 / fs
        for index, signal in enumerate(signals):
            f = relative_freq[index]
            temp = temp + signal[:] * backend.exp(-1j * 2 * backend.pi * f * t)
        from ..core import WdmSignal

        signal = WdmSignal(symbols, temp, relative_freq, center_freq, signals[0].fs, device=signals[0].device)
        return signal

    return mux_real()


class Laser:

    def __init__(self, power, lw=0):
        self.power = power
        self.lw = lw

    def __call__(self, signal):
        return self.__core(signal)

    def __core(self, signal):

        @device_selection(signal.device, True)
        def core_real(backend):
            if self.lw:
                var = 2 * backend.pi * self.lw / signal.fs
                f = backend.random.normal(scale=backend.sqrt(var), size=signal.shape)
                if len(f.shape) > 1:
                    f = np.cumsum(f, axis=1)
                else:
                    f = np.cumsum(f)
                signal[:] = backend.exp(1j * f) * signal[:]
            signal.normalize()
            power = 10**(self.power/10)/1000/signal.shape[0]
            signal[:] = backend.sqrt(power) * signal[:]
            return signal
        return core_real()


class DAC:

    def __init__(self,enob,cutoff,new_sps):
        '''
        enob:effective number of bits
        cutoff: 3-db cutoff frequency
        '''
        self.enob = enob
        self.cutoff = cutoff
        self.new_sps = new_sps

    def __core(self,signal):
        from .dsp import IdealResampler
        from .filtering import filter_signal
        signal = IdealResampler(signal.sps,new_sps=self.new_sps)(signal)
        sig_enob_noise = quantize_signal(signal, nbits=self.enob, rescale_in=True, rescale_out=True)

            # Apply 2-order bessel filter to simulate frequency response of DAC
        if self.cutoff is not None:
            filter_sig_re = filter_signal(sig_enob_noise.real, signal.fs,self.cutoff, ftype="bessel", order=2)
            filter_sig_im = filter_signal(sig_enob_noise.imag, signal.fs,self.cutoff, ftype="bessel", order=2)
            sig_enob_noise.samples = filter_sig_re + 1j * filter_sig_im
        return sig_enob_noise


    def __call__(self,signal):
        return self.__core(signal)