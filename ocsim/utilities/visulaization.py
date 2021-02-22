from dearpygui.core import *
from dearpygui.simple import *
from ocsim import Signal
from ocsim import cpu
import numpy as np
from ocsim import QamSignal,SignalSetting,PulseShaping



class Visulization:

    @staticmethod
    def scatter_callback():
        pass

    @staticmethod
    def scatterplot_dp(signal,sps,start=True):
        pol = {0:'xpol',1:'ypol'}
        with cpu(signal):
            samples = signal[:,::sps]
            with window("scatterplot_xpol", height=400, width=400, x_pos=0, y_pos=0,no_close=True):
                add_plot("constl_xpol", height=350,width=350)
            with window("scatterplot_ypol", height=400, width=400, x_pos=400, y_pos=0,no_close=True):
                add_plot("constl_ypol", height=350,width=350)
            for index,row in enumerate(samples):
                add_scatter_series(f"constl_{pol[index]}","",x=row.real.tolist(),y = row.imag.tolist(),size=1)
            if start:
                start_dearpygui()

    @staticmethod

    def psd_dp(signal,start=True):
        pol = {0: 'xpol', 1: 'ypol'}

        with cpu(signal):
            with window("psd_xpol", height=400, width=400, x_pos=0, y_pos=0,no_close=True):
                add_plot("psd_xpol##plot", height=350, width=350)
            with window("psd_ypol", height=400, width=400, x_pos=400, y_pos=0,no_close=True):
                add_plot("psd_ypol##plot", height=350, width=350)
            for index, row in enumerate(signal[:]):
                from scipy.signal import welch
                f,pxx = welch(row,fs=signal.fs,nfft=2**14,detrend=None,return_onesided=False)
                pxx = 10*np.log10(pxx)
                add_line_series(f"psd_{pol[index]}##plot", "", x=f.tolist(), y=pxx.tolist())
            if start:
                start_dearpygui()

    @staticmethod
    def summary_dp(signal):
        def scatter():
            pol = {0: 'xpol', 1: 'ypol'}
            with cpu(signal):
                samples = signal[:, ::signal.sps]

                for index, row in enumerate(samples):
                    add_scatter_series(f"constl_{pol[index]}", "", x=row.real.tolist(), y=row.imag.tolist(), size=1)

        def psd():
            pol = {0: 'xpol', 1: 'ypol'}
            for index, row in enumerate(signal[:]):
                from scipy.signal import welch
                f, pxx = welch(row, fs=signal.fs, nfft=16384, detrend=None, return_onesided=False)
                pxx = 10 * np.log10(pxx/pxx.max())
                add_line_series(f"psd_{pol[index]}##plot", "", x=f.tolist(), y=pxx.tolist(),weight=4)

        def clear():

            clear_plot("constl_xpol")
            clear_plot("constl_ypol")
            clear_plot("psd_xpol##plot")
            clear_plot("psd_ypol##plot")

        # show_debug()

        pol = {0: 'x', 1: 'y'}
        with window("primary_window"):
            pass

        with window("scatterplot_xpol", height=400, width=400, x_pos=0, y_pos=0,no_close=True):
            add_plot("constl_xpol", height=350, width=350)

        with window("scatterplot_ypol", height=400, width=400, x_pos=400, y_pos=0,no_close=True):
            add_plot("constl_ypol", height=350, width=350)

        with window("psd_xpol", height=400, width=400, x_pos=0, y_pos=400,no_close=True):
            add_plot("psd_xpol##plot", height=350, width=350)

        with window("psd_ypol", height=400, width=400, x_pos=400, y_pos=400,no_close=True):
            add_plot("psd_ypol##plot", height=350, width=350)
        scatter()
        psd()
        with window("summary", height=600, width=400, x_pos=1400, y_pos=0, no_close=True):
            add_table("signal", [], height=400, width=400)
            with cpu(signal):
                for index, row in enumerate(signal[:, :4096]):
                    add_column("signal", f"{pol[index]}-pol", [f"{item.real:.2f} + {item.imag:.2f}j" for item in row])
            add_button("constellation",callback=scatter)
            add_button("PSD",callback=psd)
            add_button("clear",callback=clear)
            # add_button("Time")

        set_main_window_size(1800, 1000)
        set_theme('Light')
        start_dearpygui(primary_window="primary_window")





signal = QamSignal(SignalSetting(center_freq=193.1e12))
shaping = PulseShaping(beta=0.2)
signal = shaping(signal)

Visulization.summary_dp(signal)
