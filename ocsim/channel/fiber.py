from dataclasses import dataclass
from ..device_manager import device_selection
import numpy as np
from scipy.constants import c


@dataclass
class FiberSetting:
    alpha_db: float = 0.2
    gamma: float = 1.3
    D: float = 16.7
    wave_length: float = 1550
    length: float = 80
    slope: float = 0
    step_length: float = None

    @property
    def alpha_linear(self):
        alphalin = self.alpha_db / (10 * np.log10(np.exp(1)))
        return alphalin

    @property
    def beta2_reference(self):
        return -self.D * (self.wave_length * 1e-12) ** 2 / 2 / np.pi / c / 1e-3

    def beta2(self, wave_length):
        '''
        :param wave_length: [m]
        :return: beta2 at wave_length [s^2/km]
        '''
        dw = 2 * np.pi * c * (1 / wave_length - 1 / (self.wave_length * 1e-9))
        return self.beta2_reference + self.beta3_reference * dw

    @property
    def beta3_reference(self):
        res = (self.wave_length * 1e-12 / 2 / np.pi / c / 1e-3) ** 2 * (
                2 * self.wave_length * 1e-12 * self.D + (
                self.wave_length * 1e-12) ** 2 * self.slope * 1e12)

        return res

    def leff(self, length):
        effective_length = 1 - np.exp(-self.alpha_linear * length)
        effective_length = effective_length / self.alpha_linear
        return effective_length



class NonlinearFiber(object):

    def __init__(self, fiber_setting,fft_backend,backend):
        self.setting = fiber_setting
        self.step_length = self.setting.step_length
        self.length = self.setting.length
        self.fft_backend = fft_backend
        self.plan = None
        self.backend = backend

    def prop(self,signal):

        wave_length = c/signal.center_freq
        nstep = self.length / self.step_length
        nstep = int(self.backend.floor(nstep))
        freq = self.backend.fft.fftfreq(signal.shape[1], 1 / signal.symbol_rate/signal.sps_in_fiber)
        freq = self.backend.asarray(freq,dtype = signal.dtype)
        omeg = 2 * self.backend.pi * freq
        D = -1j / 2 * self.setting.beta2(wave_length) * omeg ** 2
        D = self.backend.asarray(D,dtype = signal.dtype)
        N = 8 / 9 * 1j * self.setting.gamma
        atten = -self.setting.alpha_linear / 2
        last_step = self.length - self.step_length * nstep


        # dz/2
        signal[0], signal[1] = self.linear_prop(D, signal[0], signal[1], self.step_length / 2)
        # nonlinear
        signal[0], signal[1] = self.nonlinear_prop(N, signal[0], signal[1])
        # atten dz/2
        signal[0] = signal[0] * self.backend.exp(atten * self.step_length)
        signal[1] = signal[1] * self.backend.exp(atten * self.step_length)

        for _ in range(nstep-1):
            # cd
            signal[0], signal[1] = self.linear_prop(D, signal[0], signal[1], self.step_length)

            signal[0], signal[1] = self.nonlinear_prop(N, signal[0], signal[1])
            signal[0] = signal[0] * self.backend.exp(atten * self.step_length)
            signal[1] = signal[1] * self.backend.exp(atten * self.step_length)

        signal[0], signal[1] = self.linear_prop(D, signal[0], signal[1], self.step_length / 2)

        if last_step:
            last_step_eff = (1 - self.backend.exp(-self.setting.alpha_linear * last_step)) / self.setting.alpha_linear
            signal[0], signal[1] = self.linear_prop(D, signal[0], signal[1], last_step / 2)
            signal[0], signal[1] = self.nonlinear_prop(N, signal[0], signal[1], last_step_eff)
            signal[0] = signal[0] * self.backend.exp(atten * last_step)
            signal[1] = signal[1] * self.backend.exp(atten * last_step)
            signal[0], signal[1] = self.linear_prop(D, signal[0], signal[1], last_step / 2)

        return signal

    @property
    def step_length_eff(self):
        return (1 - self.backend.exp(-self.setting.alpha_linear * self.setting.step_length)) / self.setting.alpha_linear

    def nonlinear_prop(self, N, time_x, time_y, step_length=None):
        if step_length is None:
            time_x = time_x * self.backend.exp(
                N * self.step_length_eff * (self.backend.abs(time_x) ** 2 + self.backend.abs(
                    time_y) ** 2))
            time_y = time_y * self.backend.exp(
                N * self.step_length_eff * (self.backend.abs(time_x) ** 2 + self.backend.abs(time_y) ** 2))
        else:
            time_x = time_x * self.backend.exp(
                N * step_length * (self.backend.abs(time_x) ** 2 + self.backend.abs(
                    time_y) ** 2))
            time_y = time_y * self.backend.exp(
                N * step_length * (self.backend.abs(time_x) ** 2 + self.backend.abs(time_y) ** 2))

        return time_x, time_y

    def linear_prop(self, D, timex, timey, length):

        freq_x = self.fft_backend.fft.fft(timex)
        freq_y = self.fft_backend.fft.fft(timey)

        freq_x = freq_x * self.backend.exp(D * length)
        freq_y = freq_y * self.backend.exp(D * length)

        time_x = self.fft_backend.fft.ifft(freq_x)
        time_y = self.fft_backend.fft.ifft(freq_y)
        return time_x, time_y



def prop(signal,fiber_setting):

    @device_selection(signal.device,True)
    def prop_(backend):

        fiber = NonlinearFiber(fiber_setting,backend,backend)
        return fiber.prop(signal=signal)

    return prop_()