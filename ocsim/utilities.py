import matplotlib.pyplot as plt

import numpy as np
def scatterplot(signal, interval=1):
    plt.figure()
    try:
        if 'cuda' in signal.device:

            samples = np.atleast_2d(signal.samples.get())[:, ::interval]
        else:
            samples = np.atleast_2d(signal.samples[:, ::interval])
    except AttributeError:

        samples = np.copy(signal)

    with plt.style.context(['science', 'ieee', 'no-latex']):
        fig, axes = plt.subplots(1, samples.shape[0])
        axes = np.atleast_2d(axes)[0]
        pol = 0
        for ax in axes:
            xlim = [samples[pol].real.min() - 0.05, samples[pol].real.max() + 0.05]
            ylim = [samples[pol].imag.min() - 0.05, samples[pol].imag.max() + 0.05]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.scatter(samples[pol].real, samples[pol].imag,c='b',s=1)
            pol += 1
            ax.set_aspect(1)
        # viz = visdom.Visdom()
        # viz.matplot(fig)
        plt.tight_layout()
        plt.show()

