#%%
from python_library import QamSignal
from python_library import pulseshaping
from python_library import ideal_dac
from python_library import SignalSetting
import numpy as np
import cupy as cp
np.random.seed(0)
cp.random.seed(0)
# %%
signal = QamSignal(SignalSetting(193.1e12,4,2,'cpu'))
signal.to('cpu')
import numpy as np
import cupy as cp
np.random.seed(0)
cp.random.seed(0)

signal_cuda = QamSignal(SignalSetting(193.1e12,4,2,'cuda'))





# %%
signal = pulseshaping(signal,0.02)
signal_cuda = pulseshaping(signal_cuda,0.02)
# %%
signal = ideal_dac(signal,signal.sps_dsp,signal.sps_in_fiber)
signal_cuda.to('cpu')
signal_cuda = ideal_dac(signal_cuda,signal_cuda.sps_dsp,signal_cuda.sps_in_fiber)

signal.normalize()
signal_cuda.normalize()
signal[:] = np.sqrt(1/1000/2) * signal[:]
signal_cuda[:] = np.sqrt(1/1000/2) * signal_cuda[:]
# %%
signal.to('cuda')
signal_cuda.to('cuda')

from python_library import FiberSetting,prop

signal = prop(signal,FiberSetting(step_length=20/1000))
signal_cuda = prop(signal_cuda,FiberSetting(step_length=20/1000))

# %%
signal.to('cpu')
from python_library import cd_compensation

signal = cd_compensation(signal,FiberSetting(length=80),signal.symbol_rate*signal.sps_in_fiber)
# %%
signal_cuda.to('cuda')
signal_cuda = cd_compensation(signal_cuda,FiberSetting(),signal_cuda.symbol_rate * signal.sps_in_fiber)

# %%
signal = ideal_dac(signal,signal.sps_in_fiber,signal.sps_dsp)
# %%
signal = pulseshaping(signal,0.02)
# %%

from python_library.utilities import scatterplot
signal.normalize()
scatterplot(signal,2)