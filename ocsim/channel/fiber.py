import dataclasses

import numpy as np
from scipy.constants import c


@dataclasses.dataclass
class FiberSetting:
    alpha_db: float = 0.2
    gamma: float = 1.3
    length: float = 80
    D: float = 16.7
    slope: float = 0
    reference_wavelength_nm: float = 1550

    step_length: float = 20 / 1000

    @property
    def alphalin(self):
        alphalin = self.alpha_db / (10 * np.log10(np.exp(1)))
        return alphalin

    @property
    def beta2_reference(self):
        return -self.D * (self.reference_wavelength_nm * 1e-12) ** 2 / 2 / np.pi / c / 1e-3

    def beta2(self, wave_length_m):
        '''
        :param wave_length: [m]
        :return: beta2 at wave_length [s^2/km]
        '''
        dw = 2 * np.pi * c * (1 / wave_length_m - 1 / (self.reference_wavelength_nm * 1e-9))
        return self.beta2_reference + self.beta3_reference * dw

    @property
    def beta3_reference(self):
        res = (self.reference_wavelength_nm * 1e-12 / 2 / np.pi / c / 1e-3) ** 2 * (
                2 * self.reference_wavelength_nm * 1e-12 * self.D + (
                self.reference_wavelength_nm * 1e-12) ** 2 * self.slope * 1e12)

        return res

    def leff(self, length):
        '''
        :param length: the length of a fiber [km]
        :return: the effective length [km]
        '''
        effective_length = 1 - np.exp(-self.alphalin * length)
        effective_length = effective_length / self.alphalin
        return effective_length


class NonlinearFiber:

    def __init__(self, fiber_setting: FiberSetting):
        '''
            :param: kwargs:
                key: step_length
                key:gamma
        '''
        self.setting = fiber_setting

    @property
    def step_length_eff(self):
        return self.setting.leff(self.setting.step_length)

    def prop(self, signal):
        from ..device_manager import device_selection

        @device_selection(signal.device, True)
        def prop_(backend):
            nstep = self.setting.length / self.setting.step_length
            nstep = int(np.floor(nstep))
            freq = backend.fft.fftfreq(signal.shape[1], 1 / signal.fs)
            omeg = 2 * backend.pi * freq
            self.D = -1j / 2 * self.setting.beta2(c / signal.center_freq) * omeg ** 2
            N = 8 / 9 * 1j * self.setting.gamma
            atten = -self.setting.alphalin / 2
            last_step = self.setting.length - self.setting.step_length * nstep

            signal[0], signal[1] = self.linear_prop(backend, signal[0], signal[1], self.setting.step_length / 2)
            signal[0], signal[1] = self.nonlinear_prop(backend, N, signal[0], signal[1])
            signal[0] = signal[0] * backend.exp(atten * self.setting.step_length)
            signal[1] = signal[1] * backend.exp(atten * self.setting.step_length)

            for _ in range(nstep - 1):
                signal[0], signal[1] = self.linear_prop(backend, signal[0], signal[1], self.setting.step_length)

                signal[0], signal[1] = self.nonlinear_prop(backend, N, signal[0], signal[1])
                signal[0] = signal[0] * backend.exp(atten * self.setting.step_length)
                signal[1] = signal[1] * backend.exp(atten * self.setting.step_length)

            signal[0], signal[1] = self.linear_prop(backend, signal[0], signal[1], self.setting.step_length / 2)

            if last_step:
                last_step_eff = (1 - backend.exp(-self.setting.alphalin * last_step)) / self.setting.alphalin
                signal[0], signal[1] = self.linear_prop(backend, signal[0], signal[1], last_step / 2)
                signal[0], signal[1] = self.nonlinear_prop(backend, N, signal[0], signal[1], last_step_eff)
                signal[0] = signal[0] * backend.exp(atten * last_step)
                signal[1] = signal[1] * backend.exp(atten * last_step)
                signal[0], signal[1] = self.linear_prop(backend, signal[0], signal[1], last_step / 2)

            return signal

        return prop_()

    def nonlinear_prop(self, backend, N, time_x, time_y, step_length=None):
        if step_length is None:
            time_x = time_x * backend.exp(
                N * self.step_length_eff * (backend.abs(time_x) ** 2 + backend.abs(
                    time_y) ** 2))
            time_y = time_y * backend.exp(
                N * self.step_length_eff * (backend.abs(time_x) ** 2 + backend.abs(time_y) ** 2))
        else:
            time_x = time_x * backend.exp(
                N * step_length * (backend.abs(time_x) ** 2 + backend.abs(
                    time_y) ** 2))
            time_y = time_y * backend.exp(
                N * step_length * (backend.abs(time_x) ** 2 + backend.abs(time_y) ** 2))

        return time_x, time_y

    def linear_prop(self, backend, timex, timey, length):
        D = self.D
        freq_x = backend.fft.fft(timex)
        freq_y = backend.fft.fft(timey)

        freq_x = freq_x * backend.exp(D * length)
        freq_y = freq_y * backend.exp(D * length)

        time_x = backend.fft.ifft(freq_x)
        time_y = backend.fft.ifft(freq_y)
        return time_x, time_y

    def __call__(self, signal):
        return self.prop(signal)


def pmd(signal):
    pass


def sop(signal):
    pass
