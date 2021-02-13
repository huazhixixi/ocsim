from ocsim import QamSignal
from ocsim import *
import numpy as np
import cupy as cp
import tqdm


def simulate(is_wss):
    for power in tqdm.tqdm(np.arange(-3,4)):
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
            # signal = edfa(signal)
            signal[:] = np.sqrt(10**(16/10)) * signal[:]
            if is_wss:
                wss = WSS(0, 50e9, 8.8e9)
                power_ = signal.power(False)
                signal = wss(signal)
                power_2 = signal.power(False)
                signal[:] = np.sqrt(power_ / power_2) * signal[:]
        save_matfiles(signal,f'only_nli{power:.2f}',False)


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
    print(snr_meter(signal))
    return signal.normalize()

# simulate(False)
#
#
from scipy.signal import correlate
#
#
amp_corr_list = []
pha_corr_list = []
for power in [-3,-2,-1,0,1,2]:

    signal = receiver(f'only_nli{power:.2f}.mat')
    amplitude_noise = np.abs(signal[0]) - np.abs(signal.symbol[0])
    phase_noise = np.angle(signal[0]/signal.symbol[0])

    amp_corr = correlate(amplitude_noise,amplitude_noise)/amplitude_noise.shape[0]
    max_index = np.argmax(amp_corr)
    amp_corr = amp_corr[max_index+1:]
    pha_corr = correlate(phase_noise,phase_noise)/phase_noise.shape[0]
    max_index = np.argmax(pha_corr)
    pha_corr = pha_corr[max_index+1:]
    amp_corr_list.append(amp_corr)
    pha_corr_list.append(pha_corr)

from FigureManger import *




r = []
p = []
for row in amp_corr_list:
    r.append(np.abs(10*np.log10(1/(np.sum(np.abs(row[:6])))))-6)

for row in pha_corr_list:
    p.append(np.abs(10*np.log10(1/np.sum(np.abs(row[:30])))))

data = DataSetting(

    x = np.vstack([[-3,-2,-1,0,1,2]]*6),
    y = np.vstack([r,p,[30,28,26,24,22,20]])
)

layout = Layout(
    x_axis_name="Launch power [dBm]",
    y_axis_name="Value [dB]",
    legend = ("R(n)","P(n)","SNR (NLI)"),
    markers= ("o", "^", "x")
)

fig = FigureManger(data=data,layout=layout,keyword=["-3","-2","-1","0","1","2","plot the anc"])
fig.plot()