


def device_selection(device):
    def inner(func):
        def real_func(*args, **kwargs):
            if 'cuda' in device:
                print('gpu')
            else:
                print('cpu')

            # return func(signal,*args,**kwargs)

        return real_func

    return inner


