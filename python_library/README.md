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
