import numba
import numpy as np
from numba import complex128, double, boolean

lms_equalize_core_pll_type = \
    [(complex128[:, :], complex128[:, :], double, complex128[:, :], complex128[:, :], complex128[:, :],
      complex128[:, :], complex128[:, :], double, double, boolean)]


@numba.njit(lms_equalize_core_pll_type, cache=True)
def lms_equalize_core_pll(ex, ey, g, train_symbol, wxx, wyy, wxy, wyx, mu_train, mu_dd, is_train):
    symbols = np.zeros((2, ex.shape[0]), dtype=np.complex128)
    error_xpol_array = np.zeros((1, ex.shape[0]), dtype=np.float64)
    error_ypol_array = np.zeros((1, ey.shape[0]), dtype=np.float64)
    phase_error_xpol = np.zeros((1, ex.shape[0]), dtype=np.float64)
    phase_error_ypol = np.zeros((1, ex.shape[0]), dtype=np.float64)

    if is_train:
        train_symbol_xpol = train_symbol[0]
        train_symbol_ypol = train_symbol[1]

    for idx in range(len(ex)):
        xx = ex[idx][::-1]
        yy = ey[idx][::-1]
        xout_no_pll = np.sum(wxx * xx) + np.sum(wxy * yy)
        yout_no_pll = np.sum(wyx * xx) + np.sum(wyy * yy)

        symbols[0, idx] = xout_no_pll * np.exp(-1j * phase_error_xpol[0, idx])
        symbols[1, idx] = yout_no_pll * np.exp(-1j * phase_error_ypol[0, idx])

        xout = xout_no_pll * np.exp(-1j * phase_error_xpol[0, idx])
        yout = yout_no_pll * np.exp(-1j * phase_error_ypol[0, idx])

        if is_train:
            error_xpol = train_symbol_xpol[idx] - xout
            error_ypol = train_symbol_ypol[idx] - yout
            pll_error_x = np.imag(xout * np.conj(train_symbol_xpol[idx])) / np.abs(
                xout * np.conj(train_symbol_xpol[idx]))
            pll_error_y = np.imag(yout * np.conj(train_symbol_ypol[idx])) / np.abs(
                yout * np.conj(train_symbol_ypol[idx]))
            if idx < len(ex) - 1:
                phase_error_xpol[0, idx + 1] = g * pll_error_x + phase_error_xpol[0, idx]
                phase_error_ypol[0, idx + 1] = g * pll_error_y + phase_error_ypol[0, idx]

        else:
            raise NotImplementedError
            # xpol_symbol = decision(xout, constl)
            # ypol_symbol = decision(yout, constl)
            # error_xpol = xout - xpol_symbol
            # error_ypol = yout - ypol_symbol

        error_xpol_array[0, idx] = np.abs(error_xpol)
        error_ypol_array[0, idx] = np.abs(error_ypol)
        if is_train:
            mu = mu_train
        else:
            mu = mu_dd

        wxx = wxx + mu * error_xpol * np.conj(xx)
        wxy = wxy + mu * error_xpol * np.conj(yy)
        wyx = wyx + mu * error_ypol * np.conj(xx)
        wyy = wyy + mu * error_ypol * np.conj(yy)

    return symbols, wxx, wxy, wyx, wyy, error_xpol_array, error_ypol_array
