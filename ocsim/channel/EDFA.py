from scipy.constants import h, c

from ..device_manager import device_selection


class EDFA:

    def calc_noise_power(self, wavelength_m, fs):
        '''
        One pol
        '''
        ase_psd = (h * c / wavelength_m) * (self.gain_linear * 10 ** (self.nf / 10) - 1) / 2
        noise_power = ase_psd * fs
        return noise_power

    @property
    def gain_linear(self):
        return 10 ** (self.gain / 10)

    def noise_sequence(self, signal):
        noise_power = self.calc_noise_power(c / signal.center_freq, signal.fs)

        @device_selection(signal.device, True)
        def noise_sequence_real(backend):
            noise_sequence = backend.sqrt(noise_power / 2) * (
                    backend.random.randn(*signal.shape) + 1j * backend.random.randn(*signal.shape))
            return noise_sequence

        return noise_sequence_real()


class ConstantGainEDFA(EDFA):

    def __init__(self, gain, nf):
        self.gain = gain
        self.nf = nf

    def __core(self, signal):
        @device_selection(signal.device, True)
        def core_real(backend):
            noise_sequence = self.noise_sequence(signal)
            noise_power = self.calc_noise_power(c / signal.center_freq, signal.fs)
            psd = noise_power / signal.fs
            ase_12p5 = psd * 12.5e9
            signal.ase_power_12p5 += ase_12p5
            signal[:] = signal[:] * backend.sqrt(self.gain_linear)
            signal[:] = signal[:] + noise_sequence
            return signal

        return core_real()

    def __call__(self, signal):
        return self.__core(signal)
