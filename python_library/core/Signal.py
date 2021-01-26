from ..device_manager import cuda_number, device_selection
from dataclasses import dataclass


@dataclass
class SignalSetting:
    center_freq: float
    sps_in_fiber: int
    sps_dsp: int = 2
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
                 sps_in_fiber,
                 sps_dsp=2,
                 device='cpu'
                 ):

        self.samples = samples
        self.center_freq = center_freq
        self.sps_in_fiber = sps_in_fiber
        self.sps_dsp = sps_dsp
        self.device = 'cpu'

        self.to(device)
        self.make_sure_2d()

    def make_sure_2d(self):
        @device_selection(self.device,True)
        def make_sure_2d_(backend,signal):
            signal.samples = backend.atleast_2d(signal.samples)
            return signal
        make_sure_2d_(self)

    def to(self, device):

        @device_selection(device, provide_backend=True)
        def to_(backend, signal_obj):
            signal_obj.samples = backend.asarray(signal_obj.samples)
            signal_obj.device = device
            return signal_obj

        if self.device == device:
            return self
        else:
            return to_(self)

    def normalize(self):

        @device_selection(self.device, provide_backend=True)
        def normalize_(backend, signal_obj):
            factor = backend.mean(backend.abs(signal_obj[:]) ** 2, axis=-1, keedims=True)
            signal_obj[:] = signal_obj[:] / backend.sqrt(factor)
            return signal_obj

        return normalize_(self)

    def __getitem__(self, item):
        return self.samples[item]

    def __setitem__(self, key, value):
        self.samples[key] = value

    def power(self):
        @device_selection(self.device, True)
        def power_(backend, signal_obj):
            power = backend.mean(backend.abs(signal_obj) ** 2, axis=-1)
            power_dbm = 10 * backend.log10(power * 1000)

            print(f'{power[0]}:.4f W, {power[1]}:.4f W')
            print(f'{power_dbm[0]}:.4f dBm, {power[1]}:.4f dBm')

        power_(self)
    @property
    def shape(self):

        return self.samples.shape

class QamSignal(Signal):

    def __init__(self,
                 signal_setting: SignalSetting
                 ):
        import numpy as np
        from .constl import cal_symbols_qam, cal_scaling_factor_qam
        self.symbol_rate = signal_setting.symbol_rate
        self.symbol_number = signal_setting.symbol_number
        self.qam_order = signal_setting.qam_order
        self.pol_number = signal_setting.pol_number

        if signal_setting.need_init:
            self.nbits = self.symbol_number * np.log2(self.qam_order)
            self.bit_sequence = np.random.randint(0, 2, (self.pol_number, int(self.nbits)), dtype=bool)
            self.constl = cal_symbols_qam(self.qam_order) / np.sqrt(cal_scaling_factor_qam(self.qam_order))
            self.symbol = None
            self.map()
            samples = np.zeros(shape = (self.pol_number, self.symbol_number * signal_setting.sps_dsp), dtype=np.complex)
            samples[:, ::signal_setting.sps_dsp] = self.symbol
            super(QamSignal, self).__init__(samples=samples, center_freq=signal_setting.center_freq,
                                            sps_in_fiber=signal_setting.sps_in_fiber, sps_dsp=signal_setting.sps_dsp,
                                            device=signal_setting.device)

    def map(self):
        from .constl import generate_mapping, map
        _, encoding = generate_mapping(self.qam_order)
        self.symbol = map(self.bit_sequence, encoding=encoding, M=self.qam_order)


class WdmSignal(Signal):

    def __init__(self, symbols, samples, freqes, center_freq, sps_in_fiber, sps_dsp, device):
        import numpy as np
        self.symbols = symbols
        self.freq = freqes
        self.relative_freq = np.array(self.freq) - center_freq
        super(WdmSignal, self).__init__(samples=samples, center_freq=center_freq, sps_in_fiber=sps_in_fiber,
                                        sps_dsp=sps_dsp, device=device)


