
from python_library.utilities import scatterplot
from python_library import QamSignal,SignalSetting
from python_library import pulseshaping
signal_setting = SignalSetting(193.1e12,4,2,'cpu')
qam_signal = QamSignal(signal_setting)
qam_signal.to('cpu')
qam_signal.to('cuda:0')
signal = pulseshaping(qam_signal,0.02)
scatterplot(signal,2)
from python_library import ideal_dac
from python_library import prop
from python_library import FiberSetting
from python_library import NonlinearFiber

signal = ideal_dac(signal,signal.sps_dsp,signal.sps_in_fiber)
signal.normalize()
import numpy as np
signal[:] = np.sqrt (10**0/1000/2) * signal[:]
signal.power()
import time
# signal.float()
now = time.time()

signal = prop(signal,FiberSetting(step_length=100/1000))
print(time.time()-now)
signal.power()
# signal[:] = np.sqrt(10**(16/10))*signal[:]
from python_library import cd_compensation
from python_library import  LmsPll
equalizer = LmsPll(77,0.001,None,0.1,3,3)
signal = cd_compensation(signal,FiberSetting(length=80),signal.sps_in_fiber * signal.symbol_rate)
signal = ideal_dac(signal,signal.sps_in_fiber,signal.sps_dsp)
signal = pulseshaping(signal,0.02)
scatterplot(signal,2)
# signal.normalize()
# signal = equalizer(signal)