from ..device_manager import device_selection



class SignalArray(object):

    def __init__(self,samples,device):
        self.samples = samples
        self.device = device

    def __getitem__(self, item):
        return self.samples[item]

    def __setitem__(self, key, value):
        self.samples[key] = value




def normalize(signal, device):

    @device_selection(device)
    def normalize_():
        signal[:] = signal[:]

    normalize_()


def to(signal, device):

    @device_selection(device)
    def to_():
        import cupy as cp
        if 'cuda' in device:
            signal[:] = cp.asarray(signal[:], order='F')
            if hasattr(signal,'device'):
                signal.device = device
            return signal
        if 'cpu' in device:
            signal[:] = cp.asnumpy(signal[:], order='F')
            if hasattr(signal,'device'):
                signal.device = device
            return signal

    if device == signal.device:
        return signal
    else:
        to_()


normalize([1,2,3],'cuda')