# device_manager

##device.py 

function list:
    
    1. device_selection
    2. cuda_number


a decorator is defined, if cuda is used, this function will perform operation on the correct device 

Attenton:
    signal is the Signal class or its child class defined in ```signadef/Signal.py```

Examples:
```python

from python_library import device_selection
def normalize(signal,device):
    # define the real function
    
    @device_selection(device)
    def normailize_():
        # do sth to signal
        return signal
    
    normailize_()
```

##signdef

### ```Signal.py```

The base classes are defined
    
    1. Signal
    2. QamSignal

The attributes and methods:

    Signal:
       FuncList:

       AttributesList:
       
    QamSignal:
      FuncList:

      AttributesList:

Example:
```python

```

### ```Utilities.py```

    The low level operation on Signal
    
        SignalArray: a simple wrapper of numpy ndarry or cupy ndarray
        The device attributes is added for convenience

    Some low level functions:
        to(signal,device)
        normalize(signal,device)
        
Example:
```python
from python_library import SignalArray,to
import numpy as np
signal = SignalArray(np.array([1,2,3]),device='cuda:0')
signal_cuda1 = SignalArray(np.array([1,2,3]),device='cuda:1')
to(signal,'cpu')
to(signal_cuda1,'cpu')
```


    
