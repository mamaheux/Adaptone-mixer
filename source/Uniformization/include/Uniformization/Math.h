//
// Created by pascal on 9/17/19.
//

#ifndef UNIFORMIZATION_MIXER_MATH_H
#define UNIFORMIZATION_MIXER_MATH_H

#include <armadillo>
#include <math.h>
#include <cstddef>

namespace adaptone
{
    template<class T>
    inline T logSinChirp(float f1, float f2, float period, uint32_t Fs)
    {

        const std::size_t N = round(period * Fs);

        T t = arma::linspace<T>(0, period, N);
        double logf2f1 = log(f2/f1);
        return arma::sin(2 * M_PI * f1 * period / logf2f1 * (arma::exp(logf2f1 * t / period) - 1));
    }
}




#endif //UNIFORMIZATION_MIXER_MATH_H
