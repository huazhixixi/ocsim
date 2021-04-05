from dearpygui.core import *
from dearpygui.simple import *
from ..device_manager import cpu
import numpy as np

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
            fft_number = get_value("FFT_number")
            print(fft_number)
            pol = {0: 'xpol', 1: 'ypol'}
            for index, row in enumerate(signal[:]):
                from scipy.signal import welch
                f, pxx = welch(row, fs=signal.fs, nfft=fft_number, detrend=None, return_onesided=False)
                pxx = 10 * np.log10(pxx/pxx.max())
                add_line_series(f"psd_{pol[index]}##plot", "", x=f, y=pxx, weight=4)

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
        add_value("FFT_number", 16384)
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
            add_input_int("FFT_number",source="FFT_number",default_value=16384)


        set_main_window_size(1800, 1000)
        set_theme('Light')
        start_dearpygui(primary_window="primary_window")


# show_debug()

from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import dataclasses

@dataclasses.dataclass
class Layout:

    # axis setting
    xaxis_name:str
    yaxis_name:str

    # subplots setting
    nrows:int
    ncols:int

    # legend setting
    labels : [str] = ()

    # mode setting
    mode:str = "line+scatter"

    # marker setting
    markers : [str] = ()

    # color setting

    colors: [str] = ()

    default_colors:bool = True
    default_markers:bool=True

    # is_latex
    is_latex:bool = False

class TwoDimensionData:

    def __init__(self,x:[np.ndarray],y:[np.ndarray]):
        self.x = x
        self.y = y



class FigureManager:

    def __init__(self,layout:Layout,data:TwoDimensionData):
        self.layout = layout
        self.data = data
        self.plot = None

    def show(self):
        x = self.data.x
        y = self.data.y
        if self.layout.is_latex:
            style = ['science', 'ieee']
        else:
            style = ['science', 'ieee','no-latex']

        with plt.style.context(style):

            for index, row in enumerate(x):

                    if self.layout.default_markers and self.layout.default_colors:
                        self.plot = partial(plt.plot,
                                            label = self.layout.labels[index])
                    elif self.layout.default_markers and not self.layout.default_colors:
                        self.plot = partial(plt.plot,
                                            label=self.layout.labels[index],color = self.layout.colors[index])

                    elif not self.layout.default_markers and not self.layout.default_colors:
                        self.plot = partial(plt.plot,
                                            label=self.layout.labels[index],color = self.layout.colors[index],
                                            marker = self.layout.markers[index])
                    else:
                        raise NotImplementedError

                    self.plot(row,y[index])

            plt.xlabel(self.layout.xaxis_name)
            plt.ylabel(self.layout.yaxis_name)
            plt.grid()
            plt.tight_layout()
            plt.legend()
            plt.show()

    def save(self,name):
        import joblib
        joblib.dump([self.layout,self.data],name)


    @classmethod
    def load(cls,name):
        import joblib

        data = joblib.load(name)
        layout, name = data[0], data[1]

        instance = cls(layout,data)
        # instance.show()


if __name__ == '__main__':
    layout = Layout(
        xaxis_name="Giao",
        yaxis_name="heihei",
        nrows=1,
        ncols=1,
        labels=['zheshi',"eee"],
        is_latex=False
    )

    data = TwoDimensionData(
        x = np.array([[1,2,3,4],[1,2,3,4]]),
        y = np.array([[1,2,3,4],[4,5,6,7]])
    )

    fig = FigureManager(layout,data)
    fig.save("ceshi")
        # fig.show()
    fig.load("ceshi")

    fig.show()
