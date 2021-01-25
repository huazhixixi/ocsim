from python_library import Signal
import numpy as np




signal = Signal(np.array([1,2,3]),device='cuda:0',center_freq=193.1e12,sps_in_fiber=4)
signal.normalize()