import numpy as np
from ..device_manager import device_selection

def mux(signals, center_freq=None):
    freq = [signal.center_freq for signal in signals]
    device = [signal.device for signal in signals]
    length = np.diff([signal.shape[1] for signal in signals])
    fs = np.diff([signal.fs for signal in signals])

    # assert all(device)
    assert np.all(length==0)
    assert np.all(fs==0)

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
