import numpy as np
from .utilities import normalize
from .utilities import to


class Signal(object):

    def __init__(self,
                 sps_in_fiber,
                 symbol_rate,
                 center_frequency,
                 samples,
                 device='cpu',
                 sps_dsp=2,

                 ):

        self.sps_in_fiber = sps_in_fiber
        self.symbol_rate = symbol_rate
        self.center_freq = center_frequency
        self.samples = samples
        self.sps_dsp = sps_dsp
        self.device = 'cpu'
        self.to(device)


    def to(self, device):
        return to(self, device)

    def __getitem__(self, item):
        return self.samples[item]

    def __setitem__(self, key, value):
        self.samples[key] = value

    def normalize(self):
        return normalize(self, self.device)

    @classmethod
    def load_from_mat(cls, name):
        pass


class QamSignal(Signal):

    def __init__(self,
                 qam_order,
                 symbol_number,
                 sps_in_fiber,
                 symbol_rate,
                 center_frequency,
                 device='cpu',
                 sps_dsp=2,
                 is_int=True,
                 pol_dim=2
                 ):
        from .constellation import cal_symbols_qam
        from .constellation import cal_scaling_factor_qam

        self.qam_order = qam_order
        self.symbol_number = symbol_number
        self.sps_dsp = 2 if sps_dsp is None else sps_dsp
        self.pol_number = pol_dim

        self.nbits = int(self.symbol_number * np.log2(self.qam_order))
        self.bit_sequence = np.random.randint(0, 2, (self.nbits,self.pol_number), dtype=bool)
        self.bit_sequence = np.array(self.bit_sequence, order="F")
        self.constl = cal_symbols_qam(self.qam_order) / np.sqrt(cal_scaling_factor_qam(self.qam_order))
        self.constl = np.array(self.constl,order="F")

        if is_int:
            self.map()
            samples = np.zeros((self.symbol_number * self.sps_dsp, self.pol_number), order='F')
            super(QamSignal, self).__init__(sps_in_fiber=sps_in_fiber, symbol_rate=symbol_rate,
                                            center_frequency=center_frequency, device=device, samples=samples,
                                            sps_dsp=sps_dsp)

    def map(self):
        from .constellation import generate_mapping, map
        _, encoding = generate_mapping(self.qam_order)
        self.symbol = map(self.bit_sequence, encoding=encoding, M=self.qam_order)
        self.symbol = np.array(self.symbol,order='F')

    def scatterplot(self, sps=1):
        pass


if __name__ == '__main__':
    pass
