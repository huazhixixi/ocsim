from python_library import SignalArray,to
from python_library import QamSignal


signal = QamSignal(qam_order=16,symbol_number=65536,sps_in_fiber=4,symbol_rate=35e9,center_frequency=193.1e12,device='cuda:0'
                   ,sps_dsp=2)

