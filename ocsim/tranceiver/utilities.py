from ..device_manager import device_selection


def rescale_signal(signal, device, swing=1):
    """
    Rescale the (1-pol) signal to (-swing, swing).
    """

    @device_selection(device, True)
    def rescale_signal_real(backend):
        E = signal[:]

        if backend.iscomplexobj(E):
            scale_factor = backend.maximum(backend.abs(E.real).max(), backend.abs(E.imag).max())
            E = E / scale_factor * swing

        if not backend.iscomplexobj(E):
            scale_factor = backend.abs(E).max()
            E = E / scale_factor * swing

        return E

    return rescale_signal_real()


def _segment_axis(a, length, overlap, mode='cut', append_to_end=0):
    """
        Generate a new array that chops the given array along the given axis into
        overlapping frames.
        example:
        >>> segment_axis(arange(10), 4, 2)
        array([[0, 1, 2, 3],
               [2, 3, 4, 5],
               [4, 5, 6, 7],
               [6, 7, 8, 9]])
        arguments:
        a       The array to segment must be 1d-array
        length  The length of each frame
        overlap The number of array elements by which the frames should overlap
        end     What to do with the last frame, if the array is not evenly
                divisible into pieces. Options are:
                'cut'   Simply discard the extra values
                'pad'   Pad with a constant value
        append_to_end:    The value to use for end='pad'
        a new array will be returned.
    """
    if hasattr(a, 'device'):
        import cupy as np
    else:
        import numpy as np
    if a.ndim != 1:
        raise Exception("Error, input array must be 1d")
    if overlap > length:
        raise Exception("overlap cannot exceed the whole length")

    stride = length - overlap
    row = 1
    total_number = length
    while True:
        total_number = total_number + stride
        if total_number > len(a):
            break
        else:
            row = row + 1

    # 一共要分成row行
    if total_number > len(a):
        if mode == 'cut':
            b = np.zeros((row, length), dtype=np.complex128)
            is_append_to_end = False
        else:
            b = np.zeros((row + 1, length), dtype=np.complex128)
            is_append_to_end = True
    else:
        b = np.zeros((row, length), dtype=np.complex128)
        is_append_to_end = False

    index = 0
    for i in range(row):
        b[i, :] = a[index:index + length]
        index = index + stride

    if is_append_to_end:
        last = a[index:]

        b[row, 0:len(last)] = last
        b[row, len(last):] = append_to_end

    return b
