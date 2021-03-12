import numpy as np
from bitarray import bitarray


def bin2gray(value):
    """
    Convert a binary value to an gray coded value see _[1]. This also works for arrays.
    ..[1] https://en.wikipedia.org/wiki/Gray_code#Constructing_an_n-bit_Gray_code
    """
    return value ^ (value >> 1)


def cal_symbols_qam(M):
    """
    Generate the symbols on the constellation diagram for M-QAM
    """
    if np.log2(M) % 2 > 0.5:
        return cal_symbols_cross_qam(M)
    else:
        return cal_symbols_square_qam(M)


def cal_scaling_factor_qam(M):
    """
    Calculate the scaling factor for normalising MQAM symbols to 1 average Power
    """
    bits = np.log2(M)
    if not bits % 2:
        scale = 2 / 3 * (M - 1)
    else:
        symbols = cal_symbols_qam(M)
        scale = (abs(symbols) ** 2).mean()
    return scale


def cal_symbols_square_qam(M):
    """
    Generate the symbols on the constellation diagram for square M-QAM
    """
    qam = np.mgrid[-(2 * np.sqrt(M) / 2 - 1):2 * np.sqrt(
        M) / 2 - 1:1.j * np.sqrt(M), -(2 * np.sqrt(M) / 2 - 1):2 * np.sqrt(M) /
                                                               2 - 1:1.j * np.sqrt(M)]
    return (qam[0] + 1.j * qam[1]).flatten()


def cal_symbols_cross_qam(M):
    """
    Generate the symbols on the constellation diagram for non-square (cross) M-QAM
    """
    N = (np.log2(M) - 1) / 2
    s = 2 ** (N - 1)
    rect = np.mgrid[-(2 ** (N + 1) - 1):2 ** (N + 1) - 1:1.j * 2 ** (N + 1), -(
            2 ** N - 1):2 ** N - 1:1.j * 2 ** N]
    qam = rect[0] + 1.j * rect[1]
    idx1 = np.where((abs(qam.real) > 3 * s) & (abs(qam.imag) > s))
    idx2 = np.where((abs(qam.real) > 3 * s) & (abs(qam.imag) <= s))
    qam[idx1] = np.sign(qam[idx1].real) * (
            abs(qam[idx1].real) - 2 * s) + 1.j * (np.sign(qam[idx1].imag) *
                                                  (4 * s - abs(qam[idx1].imag)))
    qam[idx2] = np.sign(qam[idx2].real) * (
            4 * s - abs(qam[idx2].real)) + 1.j * (np.sign(qam[idx2].imag) *
                                                  (abs(qam[idx2].imag) + 2 * s))
    return qam.flatten()


def gray_code_qam(M):
    """
    Generate gray code map for M-QAM constellations
    """
    Nbits = int(np.log2(M))
    if Nbits % 2 == 0:
        N = Nbits // 2
        idx = np.mgrid[0:2 ** N:1, 0:2 ** N:1]
    else:
        N = (Nbits - 1) // 2
        idx = np.mgrid[0:2 ** (N + 1):1, 0:2 ** N:1]
    gidx = bin2gray(idx)
    return ((gidx[0] << N) | gidx[1]).flatten()


def generate_mapping(M, dtype=np.complex128):
    Nbits = np.log2(M)
    symbols = cal_symbols_qam(M).astype(dtype)
    scale = cal_scaling_factor_qam(M)
    symbols /= np.sqrt(scale)
    _graycode = gray_code_qam(M)
    coded_symbols = symbols[_graycode]
    bformat = "0%db" % Nbits
    encoding = dict([(symbols[i],
                      bitarray(format(_graycode[i], bformat)))
                     for i in range(len(_graycode))])
    # bitmap_mtx = generate_bitmapping_mtx(coded_symbols, cls._demodulate(coded_symbols, encoding), M, dtype=dtype)
    # return coded_symbols, _graycode, encoding, bitmap_mtx
    return coded_symbols, encoding


def map(data, encoding, M, dtype=np.complex128):
    """
    Modulate a bit sequence into QAM symbols
    Parameters
    ----------
    data     : array_like
       1D array of bits represented as bools. If the len(data)%self.M != 0 then we only encode up to the nearest divisor
    Returns
    -------
    outdata  : array_like
        1D array of complex symbol values. Normalised to energy of 1
    """
    data = np.atleast_2d(data)
    nmodes = data.shape[0]
    bitspsym = int(np.log2(M))
    Nsym = data.shape[1] // bitspsym
    # print(Nsym)
    out = np.empty((nmodes, Nsym), dtype=dtype)
    N = data.shape[1] - data.shape[1] % bitspsym
    # print(N)
    for i in range(nmodes):
        datab = bitarray()
        datab.pack(data[i, :N].tobytes())
        # the below is not really the fastest method but easy encoding/decoding is possible
        out[i, :] = np.frombuffer(b''.join(datab.decode(encoding)), dtype=dtype)
    return out
