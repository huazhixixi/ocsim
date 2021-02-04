import matplotlib.pyplot as plt

from ..core import Signal

import numpy as np


def scatterplot(signal, interval=1):
    # plt.figure()
    try:
        if 'cuda' in signal.device:

            samples = np.atleast_2d(signal.samples.get())[:, ::interval]
        else:
            samples = np.atleast_2d(signal.samples[:, ::interval])
    except AttributeError:

        samples = np.copy(signal)

    with plt.style.context(['ieee', 'science','grid', 'no-latex']):
        fig, axes = plt.subplots(1, samples.shape[0])
        axes = np.atleast_2d(axes)[0]
        pol = 0
        for ax in axes:
            xlim = [samples[pol].real.min() - 0.05, samples[pol].real.max() + 0.05]
            ylim = [samples[pol].imag.min() - 0.05, samples[pol].imag.max() + 0.05]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.scatter(samples[pol].real, samples[pol].imag, c='b', s=1)
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
    if not is_wdm:
        signal = QamSignal(SignalSetting(center_freq=center_freq,
                                         sps=sps,
                                         device=device,
                                         symbol_rate=symbol_rate,
                                         symbol_number=symbol_number,
                                         qam_order=qam_order,
                                         need_init=False,
                                         pol_number=pol_number
                                         ),symbol=symbol,samples=samples)

    else:
        raise NotImplementedError

    return signal


def save_matfiles(signal: QamSignal, file_name,is_wdm):
    from scipy.io import savemat
    device = signal.device
    signal.to('cpu')
    if '.mat' not in file_name:
        file_name = file_name + '.mat'
    if not is_wdm:
        savemat(file_name, dict(symbol=signal.symbol, samples=signal[:], center_freq=signal.center_freq,
                                sps=signal.sps, symbol_rate=signal.symbol_rate, symbol_number=signal.symbol_number,
                                qam_order=signal.qam_order, pol_number=signal.pol_number
                                ))
    else:
        raise NotImplementedError
    signal.to(device)