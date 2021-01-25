# device_manager

##device.py 

a decorator is defined, if cuda is used, this function will perform operation on the correct device 

Attentin:
    signal is the Signal class or its child class defined in ```signadef/Signal.py```

```python
from .device_manager import device_selection
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
        __init__:
        normalize:
        __getitme__:
        __setitem__:




### ```Utilities.py```

    The low level operation on Signal
    
        SignalArray: a simple wrapper of numpy ndarry or cupy ndarray
        The device attributes is added for convenience
    
    Some low level functions:
        to(signal,device)
        normalize(signal,device)