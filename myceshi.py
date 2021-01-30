import numpy as np
import cupy as cp
from ocsim import QamSignal, SignalSetting, PulseShaping, IdealResampler
from ocsim import NonlinearFiber, FiberSetting

fiber = NonlinearFiber(fiber_setting=FiberSetting())
pulse_shaping = PulseShaping(0.01)
for power in range(5):
    np.random.seed(0)
    cp.random.seed(0)
    signal = QamSignal(SignalSetting(device='cuda', center_freq=193.1e12))
    signal = pulse_shaping(signal)
    signal.to('cpu')
    dac = IdealResampler(old_sps=signal.sps, new_sps=4)
    signal = dac(signal)

    signal.to('cuda')
    signal.normalize()
    signal[:] = np.sqrt(10 ** (power / 10) / 1000 / 2) * signal[:]
    for i in range(3):
        signal = fiber(signal)
        signal[:] = np.sqrt(10 ** (16 / 10)) * signal[:]
    from ocsim import CDC

    cdc = CDC([FiberSetting()] * 3)
    signal = cdc(signal)
    signal.to('cpu')
    dac = IdealResampler(old_sps=signal.sps, new_sps=2)
    signal = dac(signal)
    signal = pulse_shaping(signal)
    signal.normalize()
    from ocsim.utilities import scatterplot

    signal.samples = signal.samples[:, ::2]
    phase = np.mean(np.angle(signal[:] / signal.symbol), axis=-1, keepdims=True)
    signal[:] = signal[:] * np.exp(-1j * phase)
    noise = signal[:] - signal.symbol
    noise = np.sum(np.mean(np.abs(noise) ** 2, axis=-1))
    print(10 * np.log10((2 - noise) / noise), )
    scatterplot(signal, 1)
