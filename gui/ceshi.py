from ocsim import PulseShaping
from ocsim import QamSignal
from ocsim import SignalSetting
from ocsim import IdealResampler
from dearpygui.core import *
from dearpygui.simple import *


def myfunciton1(signal, x1, x2, x3):
    print("myfuncition", x1, x2, x3)
    return signal

cus = {'myfunction1_custome': myfunciton1}


class Transimitter:

    def __init__(self):

        self.start_gui()

    def tx(self, sender, data):
        tx_dsp = get_value('tx_dsp_sequence')
        import json
        setting = SignalSetting(center_freq=193.1e12, sps=2, device='cpu', symbol_rate=35e9,
                                symbol_number=int(get_value('simulation_symbol_number')),
                                qam_order=16
                                )
        signal = QamSignal(signal_setting=setting)
        tx_dsp = json.loads(tx_dsp)

        function = []
        for key in tx_dsp:
            setting = tx_dsp[key]

            if 'rrc' in key:
                callable = PulseShaping(beta=setting.get('roll_off', 0.02))
                callable.order = int(key.split('#')[-1])
                function.append(callable)
            if "resample" in key:
                callable = IdealResampler(old_sps=2, new_sps=setting.get('new_sps', 4))
                callable.order = int(key.split('#')[-1])
                function.append(callable)
            if "custome" in key:
                from functools import partial
                name = key.split('#')[0]
                callable = partial(cus[name], **setting)
                callable.order = int(key.split('#')[-1])
                function.append(callable)

        function.sort(key=lambda func: func.order)
        for func in function:
            signal = func(signal)

        print(function)

    def tx_setting(self):
        with window('tx_setting_window', width=800, no_close=True):
            with group("Tx_setting", width=300):
                add_combo('modulation_format', items=['qpsk', '16-qam'], default_value='16-qam')
                add_slider_int(name='simulation_symbol_number', min_value=1024, max_value=2 ** 24, default_value=65536)
                add_slider_float('launch_power', min_value=-50, max_value=50, default_value=0)
                add_input_text("tx_dsp_sequence", default_value='{\n "rrc#1":{"roll_off":0.02},\n "resampler#2":{"new_sps":4},'
                                                                '\n "myfunction1_custome#3":{"x1":1,"x2":2,"x3":3}'
                                                                '\n}',
                               multiline=True)
                add_button('confirm', callback=self.tx)

    def start_gui(self):
        with window("main_window"):
            with group("Panel"):
                add_button("tx_setting", callback=self.tx_setting)
                add_spacing(count=5)
                add_button("link_setting")
                add_spacing(count=5)
                add_button("rx_setting")
                add_spacing(count=5)
                add_button("summary")
        start_dearpygui(primary_window="main_window")


Transimitter()
