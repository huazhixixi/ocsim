def filter_signal(signal, fs, cutoff, ftype="bessel", order=2, analog=False):
    """
    Apply an analog filter to a signal for simulating e.g. electrical bandwidth limitation
    Parameters
    ----------
    signal  : array_like
        input signal array
    fs      : float
        sampling frequency of the input signal
    cutoff  : float
        3 dB cutoff frequency of the filter
    ftype   : string, optional
        filter type can be either a bessel, butter, exp or gauss filter (default=bessel)
    order   : int
        order of the filter
    Returns
    -------
    signalout : array_like
        filtered output signal
    """
    import numpy as np
    import scipy.signal as scisig
    from ..utilities import cpu
    with cpu(signal):
        sig = np.atleast_2d(signal[:])

        Wn = cutoff * 2 * np.pi if analog else cutoff
        frmt = "ba" if analog else "sos"
        fs_in = None if analog else fs

        if ftype == "bessel":
            system = scisig.bessel(order, Wn, 'low', norm='mag', analog=analog, output=frmt, fs=fs_in)
        elif ftype == "butter":
            system = scisig.butter(order, Wn, 'low', analog=analog, output=frmt, fs=fs_in)

        if analog:
            t = np.arange(0, sig.shape[1]) * 1 / fs
            sig2 = np.zeros_like(sig)
            for i in range(sig.shape[0]):
                sig2[i] = scisig.lfilter(system[0], system[1], sig[i])

                # sig2[i] = yo.astype(sig.dtype)
        else:
            sig2 = scisig.sosfilt(system.astype(sig.dtype), sig, axis=-1)
        signal.samples = sig2
        # signal.to(device)
    return signal, system


#
# def moving_average(sig, N=3):
#     """
#     Moving average of signal
#     Parameters
#     ----------
#     sig : array_like
#         Signal for moving average
#     N: number of averaging samples
#     Returns
#     -------
#     mvg : array_like
#         Average signal of length len(sig)-n+1
#     """
#     sign = np.atleast_2d(sig)
#     ret = np.cumsum(np.insert(sign, 0,0, axis=-1), dtype=sig.dtype, axis=-1)
#     if sig.ndim == 1:
#         return ((ret[:, N:] - ret[:,:-N])/N).flatten()
#     else:

def ideal_low_filter(signal, bw):
    pass
