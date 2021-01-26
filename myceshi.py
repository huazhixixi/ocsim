#
# from python_library import QamSignal,SignalSetting
# class Simulator:
#
#     def __init__(self,config_json):
#         import json
#
#         with open(config_json) as config_json:
#             config = json.load(config_json)
#
#         signal_setting = config['signal_setting']
#
#         signal = QamSignal(signal_setting=signal_setting)
#
#
#     def config_link(self):
#         pass
#
#     def generate_signal(self):
#         pass
#
#     def simulation(self):
#         pass
#
#     def config_receiver(self):
#         pass
#
#     def save_signal(self):
#         pass


from python_library import QamSignal,SignalSetting
from python_library import pulseshaping

signal_setting = SignalSetting(193.1e12,4,2,'cpu')
qam_signal = QamSignal(signal_setting)
qam_signal.to('cpu')
qam_signal.to('cuda:0')
qam_signal.to('cpu')
qam_signal.to('cuda:1')
qam_signal.to('cpu')
signal = pulseshaping(qam_signal,0.02)


