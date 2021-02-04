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

    with plt.style.context(['science','ieee','grid','no-latex']):
        try:
            plt.psd(wdm_signal[0],NFFT=16384)
            plt.tight_layout()
            plt.savefig('/home/huazhilun/code/notes/public/wdm.png')
        except Exception:
            plt.psd(wdm_signal[0].get(), NFFT=16384)
            plt.tight_layout()
            plt.savefig('/home/huazhilun/code/notes/public/wdm.png')
        return wdm_signal

# test_mux('cuda:0')
from ocsim import Laser,QamSignal,DAC,IdealResampler,PulseShaping,ConstantGainEDFA,WSS
#
signal = QamSignal(SignalSetting(device='cuda',center_freq=193.1e12))
shaping = PulseShaping(0.02)
signal = shaping(signal)
resampler = IdealResampler(signal.sps,4)
signal = resampler(signal)
laser = Laser(1,None)
signal = laser(signal)
# signal.power()
#
# fiber = NonlinearFiber(FiberSetting())
# edfa = ConstantGainEDFA(16,5)
wss1 = WSS(0,30e9,8.8e9)
wss2 = WSS(0,30e9,13.8e9)
wss3 = WSS(0,30e9,15.8e9)
wss4 = WSS(0,30e9,20.8e9)

# signal = fiber(signal)
# signal = edfa(signal)
# power = signal.power(False)
signal = wss1(signal)
signal = wss2(signal)
signal = wss3(signal)
signal = wss4(signal)

print('hello')


from ocsim import FiberSetting

setting = FiberSetting(alpha_db=0.2,gamma=1.3,length=80,D=16.7,slope=0,reference_wavelength_nm=1550,step_length=20/1000)

# power_2 = signal.power(False)
# signal[:] = np.sqrt(power/power_2) * signal[:]
# print(signal.power())