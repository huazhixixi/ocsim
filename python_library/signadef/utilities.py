from ..device_manager import device_selection

def normalize(signal, device):
    @device_selection(device)
    def normalize_():
        signal[:] = signal[:]

    normalize_()


def to(signal, device):
    @device_selection(device)
    def to_():

        if 'cuda' in device:
            import cupy as cp
            signal.samples = cp.asarray(signal.samples, order='F')
            return signal

        if 'cpu' in device:
            import cupy as cp
            signal.samples = cp.asnumpy(signal.samples, order='F')
            return signal

    if device == signal.device:
        return signal
    else:
        to_()
