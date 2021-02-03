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
    return snr_meter(signal)

# simulate(True)


#
snrs_nonli = []
for power in (np.arange(-3,3,0.5)):
    snrs_nonli.append(receiver(f'data/nonli_power_{power:.2f}.mat'))




snrs_nli = []
for power in (np.arange(-3,3,0.5)):
    snrs_nli.append(receiver(f'data/power_{power:.2f}.mat'))

snrs_wssnli = []
for power in (np.arange(-3,3,0.5)):
    snrs_wssnli.append(receiver(f'data/wss_power_{power:.2f}.mat'))

from FigureManger import *

layout = Layout(x_axis_name='Launch Power',y_axis_name='SNR [dB]',
legend=("ASE","ASE + NLI","ASE + NLI + WSS"),markers=('o','*','x'),style=('science','ieee','no-latex'))

data = DataSetting(np.atleast_2d([np.arange(-3,3,0.5)]*3),np.atleast_2d(np.array([snrs_nonli,snrs_nli,snrs_wssnli])))
fig = FigureManger(data,layout,keyword=['ase','nli+ase',"ase+nli+wss"])
fig.plot()
fig.save("JOCN_figure")

#
#
fig.update_layout(dict(style=('science', 'ieee', 'grid', 'no-latex')))
fig.update_layout(dict(legend=('ASE', "NLI+ASE", "NLI+ASE+WSS")))
