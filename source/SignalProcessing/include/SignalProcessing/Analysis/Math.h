#ifndef SIGNAL_PROCESSING_ANALYSIS_MATH_H
#define SIGNAL_PROCESSING_ANALYSIS_MATH_H

#include <armadillo>

#include <cstddef>

namespace adaptone
{
    template<class T>
    inline T hamming(std::size_t length)
    {
        const std::size_t N = length - 1;
        T n = arma::regspace<T>(0, N);
        return 0.54 - 0.46 * arma::cos(2 * M_PI * n / N);
    }
}

#endif
