//
// Created by lun on 2021/1/24.
//

#include "dsp.h"
#include <armadillo>
#include <cassert>



arma::Mat <arma::cx_double>
lms_pll_pll (

         double lr_train,
         double lr_dd,
         arma::uword sps,
         arma::uword tap_number,
         arma::Mat <arma::cx_double> train_symbol,
         arma::Mat <arma::cx_double> xpol_samples,
         arma::Mat <arma::cx_double> ypol_sample

         )

{
    //xx,xy,yx,yy
   DSP::CxMat taps {tap_number,4,arma::fill::zeros};
   taps.at(static_cast<arma::uword> (tap_number/2),0) = 1;
   taps.at(static_cast<arma::uword> (tap_number/2),3) = 1;



   return res;
}