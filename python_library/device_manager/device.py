

def cuda_number(device):
    assert 'cuda' in device
    if device == 'cuda':
        return 0
    else:
        return int(device.split(':')[-1])


def device_selection(device):
    def inner(func):
        def real_func(*args, **kwargs):
            if 'cuda' in device:
                import cupy as cp
                with cp.cuda.Device(cuda_number(device)):
                    return func(*args,**kwargs)
            else:
                return func(*args,**kwargs)

            # return func(signal,*args,**kwargs)

        return real_func

    return inner


