# device_manager

##device_manager dir 

### device.py

function list:
    
    1. device_selection
    2. cuda_number


a decorator device_selection is defined, if cuda is used, this function will perform operation on the correct device 

#### device_seltecion(device,provide_backend:bool)
if a function is decortated by this decorator, args is a,b,c,d

if provided_backend is True, the following is performed:

    func(backend,a,b,c,d)

The backend is set in the decorator, is device is 'cuda', the backend will be 
set to cupy, and if device is 'cpu', the backend will be set to numpy

Examples:

```python

from ocsim import device_selection


def normalize(signal, device):
    # define the real function
    @device_selection(device, provide_backend=True)
    # real_func(*args,**kwargs)
    # if provid_backend:
    #   func(backend,*args,**kwargs)
    # else:
    #   func(*args,**kwargs)
    def normailize_(backend, args):
        # do sth to signal
        return 2

    normailize_(1)  # real_func(1)----> normailize(backend,1)
```
## core dir:
    1. constl.py
    2. Signal.py

### constl.py
#TO BE ADDED

### Signal.py
Two classes are defined here:
    
    Signal: The baseclass of all Signal

    QamSignal: The class represents QamSignal

define a QamSignal object

```python
from ocsim import QamSignal, SignalSetting

signal_setting = SignalSetting(
    center_freq=193.1e12, sps_in_fiber=4,
    device='cuda:0', symbol_rate=35e9,
    sps_dsp=2
)

signal = QamSignal(signal_setting=signal_setting)

signal.normalize()  # inplace normalize the signal
signal.to('cuda:1')  # Move signal to NVIDIA GRAPHIC CARD
signal.power()  # print the signal power in W and dBm

```
## Tranceiver dir
### dsp.py
Define the function at Tx and Rx

Tx: rrc pulse shaping 

Rx: CDC, matched_filter, LMS_PLL and superscalaer
### instruments.py
Define the electrical and optical instruments, at current state, Laser and Mux are implemented

### numba_backend.py

Define the low-level lms_pll function using numba LLVM compiler to speed up

#### TODO: implement the CPP backend for LMS_PLL