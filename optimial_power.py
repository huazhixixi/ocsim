from ocsim import *
import numpy as np
import cupy as cp
import tqdm


def simulate(is_wss):
    for power in tqdm.tqdm(np.arange(-3,3,0.5)):
        cp.random.seed(0)
        np.random.seed(0)
        signal = QamSignal(SignalSetting(device='cuda:0', center_freq=193.1e12))
        shaping = PulseShaping(0.02)
        signal = shaping(signal)
        resampler = IdealResampler(signal.sps, 4)
        signal = resampler(signal)
        laser = Laser(power, None)
        signal = laser(signal)
        for _ in range(15):
            fiber = NonlinearFiber(FiberSetting(step_length=100/1000))
            edfa = ConstantGainEDFA(16, 5)

            signal = fiber(signal)
            signal = edfa(signal)
            if is_wss:
                wss = WSS(0, 50e9, 8.8e9)
                power_ = signal.power(False)
                signal = wss(signal)
                power_2 = signal.power(False)
                signal[:] = np.sqrt(power_ / power_2) * signal[:]
        save_matfiles(signal,f'wss_power_{power:.2f}')


def receiver(file_name):
    signal = read_matfiles(file_name)
    cdc = CDC(fiber_setting=FiberSetting(length=15*80))
    matched_filter = PulseShaping(0.02)
    resample = IdealResampler(signal.sps,2)

    signal = cdc(signal)
    signal = resample(signal)
    signal = matched_filter(signal)

    phase = np.angle(np.mean(signal[:,::2]/signal.symbol,axis=-1,keepdims=True))
    signal[:] = signal[:] * np.exp(-1j*phase)
    signal.samples = signal[:,::2]
    scatterplot(signal,2,True)
    return snr_meter(signal)

#
signal = QamSignal(SignalSetting(device='cuda:0', center_freq=193.1e12))
# scatterplot(signal.symbol,1,False,size=3)
shaping = PulseShaping(0.02)
signal = shaping(signal)
resampler = IdealResampler(signal.sps, 4)
signal = resampler(signal)
laser = Laser(2, None,None)
signal = laser(signal)
for _ in tqdm.tqdm(range(15)):
    fiber = NonlinearFiber(FiberSetting(step_length=80,gamma=0))
    edfa = ConstantGainEDFA(16, 5)

    signal = fiber(signal)
    signal = edfa(signal)
    # signal[:] = np.sqrt(10**(16/10)) * signal[:]

save_matfiles(signal,'ase_2dbm',False)
# print(receiver('no_ase_3dbm.mat'))
print(receiver('ase_2dbm.mat'))