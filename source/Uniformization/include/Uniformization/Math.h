#ifndef UNIFORMIZATION_MATH_H
#define UNIFORMIZATION_MATH_H

#include <SignalProcessing/Utils/Math.h>
#include <armadillo>
#include <cmath>
#include <cstddef>

namespace adaptone
{
    arma::vec linearRegression(const arma::vec& y, const arma::mat& X);

    double computeRelativePositionsFromDistances(const arma::mat& distanceMat,
        std::size_t iterationCount,
        std::size_t tryCount,
        std::size_t thermalIterationCount,
        double alpha,
        double epsilonTotalDistanceError,
        double epsilonDeltaTotalDistanceError,
        std::size_t countThreshold,
        std::size_t dimension,
        arma::mat& setAPositionMat,
        arma::mat& setBPositionMat);

    void rotateSetAroundVec3D(arma::mat& set, arma::vec u, double angle);

    void rotateSet2D(arma::mat& set, double angle);

    double findSetAngle2D(const arma::mat& set);

    arma::vec averageFrequencyBand(const arma::vec& x, const arma::vec& centerFrequencies, const size_t fs);

    inline arma::vec correlation(const arma::vec& A, const arma::vec& B)
    {
        return arma::conv(A, arma::reverse(B));
    }

    inline size_t findDelay(const arma::vec& A, const arma::vec& B)
    {
        return arma::index_max(correlation(A,B)) - (A.size() + B.size() - 2) / 2;
    }

    template<class T>
    inline T logSinChirp(double f1, double f2, double period, std::size_t fs)
    {
        const std::size_t N = round(period * fs);

        T t = arma::linspace<T>(0, period, N);
        double logf2f1 = log(f2/f1);
        return arma::sin(2 * M_PI * f1 * period / logf2f1 * (arma::exp(logf2f1 * t / period) - 1));
    }

    inline void moveSet(arma::mat& set, const arma::vec& offset)
    {
        for (std::size_t i = 0; i < set.n_cols; i++)
        {
            set.col(i) += offset(i);
        }
    }

    inline arma::vec getSetCentroid(const arma::mat& set)
    {
        return  arma::conv_to<arma::vec>::from(arma::mean(set, 0));
    }
}

#endif
