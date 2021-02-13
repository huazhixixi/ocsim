import matplotlib.pyplot as plt
from DensityPlot import density2d

from ..core import Signal

import numpy as np


def scatterplot(signal, interval=1, is_density=False, size=1):
    try:
        if 'cuda' in signal.device:

            samples = np.atleast_2d(signal.samples.get())[:, ::interval]
        else:
            samples = np.atleast_2d(signal.samples[:, ::interval])
    except AttributeError:

        samples = np.copy(signal)

    with plt.style.context(['ieee', 'science', 'grid', 'no-latex']):
        fig, axes = plt.subplots(1, samples.shape[0])
        axes = np.atleast_2d(axes)[0]
        pol = 0
        xlim = [samples[pol].real.min() - 0.005, samples[pol].real.max() + 0.005]
        ylim = [samples[pol].imag.min() - 0.005, samples[pol].imag.max() + 0.005]
        for ax in axes:

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            if not is_density:
                ax.scatter(samples[pol].real, samples[pol].imag, c='b', s=size)
            else:
                density2d(x=samples[pol].real, y=samples[pol].imag, bins=500, ax=ax, s=size)
            pol += 1
            ax.set_aspect(1)
        # viz = visdom.Visdom()
        # viz.matplot(fig)
        plt.tight_layout()
        plt.show()


def snr_meter(signal: Signal):
    signal.normalize()
    assert signal.shape == signal.symbol.shape
    noise = signal[:] - signal.symbol
    noise_power = np.sum(np.mean(np.abs(noise[:]) ** 2, axis=-1))
    return 10 * np.log10((2 - noise_power) / noise_power)


from ..core import QamSignal, SignalSetting


def read_matfiles(file_name, is_wdm=False, device='cpu'):
    from scipy.io import loadmat
    from mat73 import loadmat as loadmat73

    try:
        data = loadmat(file_name)
    except NotImplementedError:
        data = loadmat73(file_name)

    symbol = data['symbol']
    samples = data['samples']
    center_freq = data['center_freq'][0, 0]
    sps = data['sps'][0, 0]
    symbol_rate = data['symbol_rate'][0, 0]
    symbol_number = data["symbol_number"][0, 0]
    qam_order = data['qam_order'][0, 0]
    pol_number = data['pol_number'][0, 0]
    ase_power_12p5 = data['ase_power_12p5'][0,0]
    signal_power = data['signal_power'][0,0]
    if not is_wdm:
        signal = QamSignal(SignalSetting(center_freq=center_freq,
                                         sps=sps,
                                         device=device,
                                         symbol_rate=symbol_rate,
                                         symbol_number=symbol_number,
                                         qam_order=qam_order,
                                         need_init=False,
                                         pol_number=pol_number
                                         ), symbol=symbol, samples=samples)
        signal.signal_power = signal_power
        signal.ase_power_12p5 = ase_power_12p5
    else:
        raise NotImplementedError

    return signal


def save_matfiles(signal: QamSignal, file_name, is_wdm):
    from scipy.io import savemat
    device = signal.device
    signal.to('cpu')
    if '.mat' not in file_name:
        file_name = file_name + '.mat'
    if not is_wdm:
        savemat(file_name, dict(symbol=signal.symbol, samples=signal[:], center_freq=signal.center_freq,
                                sps=signal.sps, symbol_rate=signal.symbol_rate, symbol_number=signal.symbol_number,
                                qam_order=signal.qam_order, pol_number=signal.pol_number,
                                ase_power_12p5=signal.ase_power_12p5, signal_power=signal.signal_power
                                ))
    else:
        raise NotImplementedError
    signal.to(device)

from contextlib import contextmanager

@contextmanager
def cpu(signal):
    original_device = signal.device
    signal.to('cpu')
    yield signal
    signal.to(original_device)

@contextmanager
def cuda(signal,cuda_number):
    original_device = signal.device
    signal.to(f'cuda:cuda_number')
    yield signal
    signal.to(original_device)


class Transimitter:

    def __init__(self,signal_setting,dac_sps,beta,laser_power_dbm):
        self.signal_setting = signal_setting
        self.dac_sps = dac_sps
        self.beta = beta
        self.laser_power = laser_power_dbm

    def prop(self,dsp_modules=None):

        signal = QamSignal(self.signal_setting)
        if dsp_modules is None:
            dsp_modules = []
            from ..tranceiver import Laser,PulseShaping,IdealResampler
            dsp_modules.extend([PulseShaping(self.beta),IdealResampler(signal.sps,self.dac_sps),Laser(self.laser_power)])
        for module in dsp_modules:
            signal = module(signal)

        return signal