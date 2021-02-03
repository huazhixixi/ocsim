from ..device_manager import device_selection

def rescale_signal(signal,device,swing=1):
    """
    Rescale the (1-pol) signal to (-swing, swing).
    """
    @device_selection(device,True)
    def rescale_signal_real(backend):
        E = signal[:]

        if backend.iscomplexobj(E):
            scale_factor = backend.maximum(backend.abs(E.real).max(), backend.abs(E.imag).max())
            E = E/scale_factor * swing

        if not backend.iscomplexobj(E):
            scale_factor = backend.abs(E).max()
            E = E/ scale_factor * swing

        return E
    return rescale_signal_real()