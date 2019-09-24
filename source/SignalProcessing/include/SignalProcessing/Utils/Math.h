#ifndef SIGNAL_PROCESSING_UTILS_MATH_H
#define SIGNAL_PROCESSING_UTILS_MATH_H

#include <armadillo>
#include <fftw3.h>

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

    inline void fft(arma::cx_vec& x, arma::cx_vec& y)
    {
        if (y.n_elem != x.n_elem)
        {
            y = arma::zeros<arma::cx_vec>(x.n_elem);
        }

        fftw_plan plan = fftw_plan_dft_1d(x.n_elem,
            reinterpret_cast<fftw_complex*>(x.memptr()),
            reinterpret_cast<fftw_complex*>(y.memptr()),
            FFTW_FORWARD,
            FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }

    inline void ifft(arma::cx_vec& x, arma::cx_vec& y)
    {
        if (y.n_elem != x.n_elem)
        {
            y = arma::zeros<arma::cx_vec>(x.n_elem);
        }

        fftw_plan plan = fftw_plan_dft_1d(x.n_elem,
            reinterpret_cast<fftw_complex*>(x.memptr()),
            reinterpret_cast<fftw_complex*>(y.memptr()),
            FFTW_BACKWARD,
            FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);

        y /= y.n_elem;
    }

    inline void hilbert(arma::cx_vec& x, arma::cx_vec& y)
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

        arma::cx_vec xFreq;
        fft(x, xFreq);
        xFreq(arma::span(1, limit1 - 1)) *= 2;
        xFreq(arma::span(limit2, x.n_elem - 1)) *= 0;
        ifft(xFreq, y);
    }

    inline void interpolateWithNaNRemoval(const arma::vec& x, const arma::vec& y, const arma::vec& xx, arma::vec& yy)
    {
        arma::interp1(x, y, xx, yy, "*linear");

        std::size_t notNaNIndex = 0;
        while (notNaNIndex < yy.n_elem && std::isnan(yy(notNaNIndex)))
        {
            notNaNIndex++;
        }
        for (std::size_t i = 0; i < notNaNIndex; i++)
        {
            yy(i) = yy(notNaNIndex);
        }

        notNaNIndex = yy.n_elem - 1;
        while (notNaNIndex > 0 && std::isnan(yy(notNaNIndex)))
        {
            notNaNIndex--;
        }
        for (std::size_t i = yy.n_elem - 1; i > notNaNIndex; i--)
        {
            yy(i) = yy(notNaNIndex);
        }
    }
}

#endif
