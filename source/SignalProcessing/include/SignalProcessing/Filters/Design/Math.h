#ifndef SIGNAL_PROCESSING_FILTERS_DESIGN_MATH_H
#define SIGNAL_PROCESSING_FILTERS_DESIGN_MATH_H

#include <armadillo>

namespace adaptone
{
    inline void hilbert(const arma::vec& x, arma::cx_vec& y)
    {
        std::size_t limit1;
        std::size_t limit2;

        if (x.n_elem % 2 == 0)
        {
            limit1 = x.n_elem / 2;
            limit2 = limit1 + 1;
        }
        else
        {
            limit1 = (x.n_elem + 1) / 2;
            limit2 = limit1;
        }

        arma::cx_vec xFreq = arma::fft(x);
        xFreq(arma::span(1, limit1 - 1)) *= 2;
        xFreq(arma::span(limit2, x.n_elem - 1)) *= 0;
        y = arma::ifft(xFreq);
    }
}

#endif
