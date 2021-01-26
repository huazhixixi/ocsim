import numba
import numpy as np
from numba import double, complex128, int64
from numpy import imag


# from numba import prange

@numba.njit(complex128[:,:](complex128[:,:],complex128[:,:],double,double,int64,int64,double,complex128[:,:],complex128[:,:]),
            cache=True,fastmath=True)
def lms_pll(
            samples_xpol,
            samples_ypol,
            lr_train,
            lr_dd,
            train_loop_number,
            total_loop_number,
            g,
            train_symbol,
            taps):

    symbol_number = samples_xpol.shape[0]
    train_symbol_xpol = train_symbol[0]
    train_symbol_ypol = train_symbol[1]
    symbols = np.zeros((2, symbol_number), dtype=np.complex128)
    phase_error_xpol = np.zeros((1, symbol_number), dtype=np.float64)
    phase_error_ypol = np.zeros((1, symbol_number), dtype=np.float64)
    wxx = taps[0]
    wxy = taps[1]
    wyx = taps[2]
    wyy = taps[3]
    for loop_index in range(total_loop_number):
        for idx in range(symbol_number):
            xx = samples_xpol[idx, ::-1]
            yy = samples_ypol[idx, ::-1]
            xout_no_pll = np.sum(wxx*xx) + np.sum(wxy*yy)
            yout_no_pll = np.sum(wyx*xx) + np.sum(wyy*yy)

            symbols[0, idx] = xout_no_pll * np.exp(-1j * phase_error_xpol[0, idx])
            symbols[1, idx] = yout_no_pll * np.exp(-1j * phase_error_ypol[0, idx])

            xout = xout_no_pll * np.exp(-1j * phase_error_xpol[0, idx])
            yout = yout_no_pll * np.exp(-1j * phase_error_ypol[0, idx])

            if train_loop_number:
                error_xpol = train_symbol_xpol[idx] - xout
                error_ypol = train_symbol_ypol[idx] - yout

                pll_error_x = imag(xout * np.conj(train_symbol_xpol[idx])) / np.abs(xout * np.conj(train_symbol_xpol[idx]))
                pll_error_y = imag(yout * np.conj(train_symbol_ypol[idx])) / np.abs(yout * np.conj(train_symbol_ypol[idx]))
                if idx < symbol_number - 1:
                    phase_error_xpol[0, idx + 1] = g * pll_error_x + phase_error_xpol[0, idx]
                    phase_error_ypol[0, idx + 1] = g * pll_error_y + phase_error_ypol[0, idx]
            else:
                raise NotImplementedError

            if train_loop_number:
                wxx += lr_train * error_xpol * np.conj(xx * np.exp(-1j * phase_error_xpol[0, idx]))
                wxy += lr_train * error_xpol * np.conj(yy * np.exp(-1j * phase_error_ypol[0, idx]))

                wyx += lr_train * error_ypol * np.conj(xx * np.exp(-1j * phase_error_xpol[0, idx]))
                wyy += lr_train * error_ypol * np.conj(yy * np.exp(-1j * phase_error_ypol[0, idx]))

        train_loop_number -= 1
    return symbols

@numba.njit(complex128[:](complex128[:],complex128[:]),cache=True,fastmath=False)
def decision(constl,received_symbol):
    shape = received_symbol.shape
    # print(len(received_symbol))
    decision_symbol = np.zeros(shape,dtype=np.complex128)

    for idx,symbol in enumerate(received_symbol):
        # print(idx,',')
        index = np.argmin(np.abs(symbol - constl))
        decision_symbol[idx] = constl[index]

    return decision_symbol
