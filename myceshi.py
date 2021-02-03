import numpy as np
import cupy as cp
from ocsim import QamSignal, SignalSetting, PulseShaping, IdealResampler
from ocsim import NonlinearFiber, FiberSetting
from ocsim import rrcos_time

def test_fiber():
    fiber = NonlinearFiber(fiber_setting=FiberSetting())
    pulse_shaping = PulseShaping(0.2)
    for power in range(1):
        np.random.seed(0)
        cp.random.seed(0)
        signal = QamSignal(SignalSetting(device='cuda', center_freq=193.1e12))
        signal = pulse_shaping(signal)
        # signal.to('cpu')
        dac = IdealResampler(old_sps=signal.sps, new_sps=4)
        signal = dac(signal)

        # signal.to('cuda')
        signal.normalize()
        signal[:] = np.sqrt(10 ** (power / 10) / 1000 / 2) * signal[:]
        from ocsim import save_matfiles
        save_matfiles(signal,'before_prop')
        for i in range(3):
            import  time
            now = time.time()
            signal = fiber(signal)
            print(time.time()-now)
            signal[:] = np.sqrt(10 ** (16 / 10)) * signal[:]
        save_matfiles(signal,'after_prop')
        # from ocsim import CDC
        #
        # cdc = CDC([FiberSetting()] * 3)
        # # signal = cdc(signal)
        # # signal.to('cpu')
        # dac = IdealResampler(old_sps=signal.sps, new_sps=2)
        # signal = dac(signal)
        # signal = pulse_shaping(signal)
        # signal.normalize()
        # from ocsim.utilities import scatterplot
        # signal.to('cpu')
        # signal.samples = signal.samples[:, ::2]
        # signal.samples = signal.samples[:,1024:-1024]
        # signal.symbol = signal.symbol[:,1024:-1024]
        # phase = np.mean(np.angle(signal[:] / signal.symbol), axis=-1, keepdims=True)
        # signal[:] = signal[:] * np.exp(-1j * phase)
        # signal.normalize()
        # noise = signal[:] - signal.symbol
        # noise = np.sum(np.mean(np.abs(noise) ** 2, axis=-1))
        # print(10 * np.log10((2 - noise) / noise), )
        # scatterplot(signal, 1)

def test_rrc():
    signal = QamSignal(SignalSetting(device='cuda', center_freq=193.1e12))
    pulse_shaping = PulseShaping(0.2)

    signal = pulse_shaping(signal)
    dac = IdealResampler(old_sps=signal.sps, new_sps=4)
    signal = dac(signal)

    # signal.to('cuda')
    signal.normalize()
    from ocsim import save_matfiles

    dac = IdealResampler(old_sps=signal.sps, new_sps=2)
    signal = dac(signal)
    signal = pulse_shaping(signal)
    signal.normalize()
    from ocsim.utilities import scatterplot
    signal.to('cpu')
    signal.samples = signal.samples[:, ::2]
    from ocsim import snr_meter
    print(snr_meter(signal))

    # signal.samples = signal.samples[:,1024:-1024]
    # signal.symbol = signal.symbol[:,1024:-1024]
    # phase = np.mean(np.angle(signal[:] / signal.symbol), axis=-1, keepdims=True)
    # signal[:] = signal[:] * np.exp(-1j * phase)
    # signal.normalize()
    # noise = signal[:] - signal.symbol
    # noise = np.sum(np.mean(np.abs(noise) ** 2, axis=-1))
    # print(10 * np.log10((2 - noise) / noise), )
    scatterplot(signal, 1)

def test_mux(device):
    np.random.seed(0)
    cp.random.seed(0)
    pulse_shaping = PulseShaping(0.2)

    sps_in_fiber = 16
    signals = []
    for i in range(3):
        signal = QamSignal(SignalSetting(device=device, center_freq=193.1e12+i*50e9))
        signal = pulse_shaping(signal)
        # signal.to('cpu')
        dac = IdealResampler(old_sps=signal.sps, new_sps=sps_in_fiber)
        signal = dac(signal)
        signals.append(signal)

    from ocsim import mux
    wdm_signal = mux(signals)

    from ocsim import save_matfiles
    import matplotlib.pyplot as plt
    try:
        plt.psd(wdm_signal[0])
        plt.show()
    except Exception:
        plt.psd(wdm_signal[0].get())
        plt.show()
    return wdm_signal

from ocsim import Laser,QamSignal,DAC,IdealResampler,PulseShaping

signal = QamSignal(SignalSetting(device='cuda',center_freq=193.1e12))
shaping = PulseShaping(0.02)
signal = shaping(signal)
# resampler = IdealResampler(signal.sps,4)
dac = DAC(6,None,4)
signal = dac(signal)

laser = Laser(1,100e3)
signal = laser(signal)