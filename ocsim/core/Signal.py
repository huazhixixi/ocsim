from dataclasses import dataclass

from ..device_manager import device_selection
import numpy as np

@dataclass
class SignalSetting:
    center_freq: float
    sps: int = 2
    device: str = 'cpu'
    #####QAM Signal###
    symbol_rate: float = 35e9
    symbol_number: int = 65536
    qam_order: int = 16
    need_init: bool = True
    pol_number: int = 2


class Signal(object):

    def __init__(self,
                 samples,
                 center_freq,
                 sps=2,
                 device='cpu'
                 ):

        self.samples = samples
        self.center_freq = center_freq
        self.sps = sps
        self.device = 'cpu'
        self.ase_power_12p5 = 0  # set in EDFA
        self.signal_power = None  # set in Laser
        self.to(device)
        self.make_sure_2d()

    def make_sure_2d(self):
        @device_selection(self.device, True)
        def make_sure_2d_(backend, signal):
            signal.samples = backend.atleast_2d(signal.samples)
            return signal

        make_sure_2d_(self)

    def cuda(self, device):
        if device == self.device:
            return
        import cupy as cp
        from ..device_manager import cuda_number
        with cp.cuda.Device(cuda_number(device)):
            self.samples = cp.asarray(self.samples)
        self.device = device
        return self

    def cpu(self, device):
        if self.device == device:
            return self
        else:
            self.samples = self.samples.get()
            self.device = device
            return self

    def to(self, device):
        if 'cuda' in device:
            return self.cuda(device=device)
        if 'cpu' in device:
            return self.cpu(device=device)

    @property
    def dtype(self):
        return self.samples.dtype

    def normalize(self):
        @device_selection(self.device, provide_backend=True)
        def normalize_(backend, signal_obj):
            factor = backend.mean(backend.abs(
                signal_obj[:]) ** 2, axis=-1, keepdims=True)
            signal_obj[:] = signal_obj[:] / backend.sqrt(factor)
            return signal_obj

        return normalize_(self)

    def __getitem__(self, item):
        return self.samples[item]

    def __setitem__(self, key, value):
        self.samples[key] = value

    def power(self, veborse=True):
        import numpy as np

        @device_selection(self.device, True)
        def power_(backend, signal_obj):
            power = backend.mean(backend.abs(signal_obj[:]) ** 2, axis=-1)
            try:
                power = power.get()
            except AttributeError:
                pass
            if veborse:
                print(f'total:{10 * np.log10(1000 * power.sum()):.4} dBm')
            return power.sum()

        return power_(self)

    @property
    def shape(self):
        return self.samples.shape

    def float(self):
        @device_selection(self.device, True)
        def float_(backend, signal_obj):
            signal_obj.samples = backend.asarray(
                signal_obj.samples, dtype=backend.complex64)
            return signal_obj

        return float_(self)

    @property
    def fs(self):
        return self.symbol_rate * self.sps

    @property
    def real(self):
        return self.samples.real

    @property
    def imag(self):
        return self.samples.imag

    def downsample(self, factor):
        self.samples = self.samples[:, ::factor]

    def summary(self):
        string1 = str(id(self))
        print(string1)
        from prettytable import PrettyTable
        information = PrettyTable()
        information.field_names = ["symbol_rate [GHz]",
                                   "symbol_length",
                                   "sps", "fs[GHz]", "mf", "center_freq[THz]", "power[dBm]"]
        information.add_row([self.symbol_rate/1e9, self.symbol_number, self.sps,
                             self.fs /1e9, self.qam_order, self.center_freq/1e12,
                             f"{10*np.log10(self.power(False)*1000) :.4}"
                             ])

        print(information)


class QamSignal(Signal):

    def __init__(self,
                 signal_setting: SignalSetting,
                 samples=None,
                 symbol=None
                 ):
        import numpy as np
        from .constl import cal_symbols_qam, cal_scaling_factor_qam
        self.symbol_rate = signal_setting.symbol_rate
        self.symbol_number = signal_setting.symbol_number
        self.qam_order = signal_setting.qam_order
        self.pol_number = signal_setting.pol_number
        self.symbol = None

        if signal_setting.need_init:
            self.nbits = self.symbol_number * np.log2(self.qam_order)
            self.bit_sequence = np.random.randint(
                0, 2, (self.pol_number, int(self.nbits)), dtype=bool)
            self.constl = cal_symbols_qam(
                self.qam_order) / np.sqrt(cal_scaling_factor_qam(self.qam_order))
            self.map()
            samples = np.zeros(shape=(
                self.pol_number, self.symbol_number * signal_setting.sps), dtype=np.complex)
            samples[:, ::signal_setting.sps] = self.symbol
            super(QamSignal, self).__init__(samples=samples, center_freq=signal_setting.center_freq,
                                            sps=signal_setting.sps,
                                            device=signal_setting.device)
        else:
            assert symbol is not None
            assert samples is not None
            self.symbol = symbol
            super(QamSignal, self).__init__(samples=samples, center_freq=signal_setting.center_freq,
                                            sps=signal_setting.sps,
                                            device=signal_setting.device)

    def map(self):
        from .constl import generate_mapping, map
        _, encoding = generate_mapping(self.qam_order)
        self.symbol = map(self.bit_sequence,
                          encoding=encoding, M=self.qam_order)


class WdmSignal(Signal):

    def __init__(self, symbols, samples, freqes, center_freq, fs, device):
        import numpy as np
        self.symbols = symbols
        self.freq = freqes
        self.relative_freq = np.array(self.freq) - center_freq
        self._fs = fs
        super(WdmSignal, self).__init__(samples=samples, center_freq=center_freq,
                                        sps=None, device=device)

    @property
    def fs(self):
        return self.fs

    @fs.setter
    def fs(self, value):
        self._fs = value
