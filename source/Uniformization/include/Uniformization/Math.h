#ifndef UNIFORMIZATION_MATH_H
#define UNIFORMIZATION_MATH_H

#include <armadillo>
#include <cmath>
#include <cstddef>

namespace adaptone
{
    arma::colvec linearRegression(const arma::vec& y, const arma::mat& X);

    double computeRelativePositionsFromDistances(const arma::mat& distanceMat, int iterationCount, int tryCount,
        int thermalIterationCount, double alpha, double epsilonTotalDistanceError, double epsilonDeltaTotalDistanceError,
        int countThreshold, int dimension, arma::mat& setAPositionMat, arma::mat& setBPositionMat);

    void rotateSetAroundVec3D(arma::mat& set, arma::vec u, double angle);

    void rotateSet2D(arma::mat& set, double angle);

    double findSetAngle2D(const arma::mat& set);

    template<class T>
    inline T logSinChirp(double f1, double f2, double period, uint32_t fs)
    {
        const std::size_t N = round(period * fs);

        T t = arma::linspace<T>(0, period, N);
        double logf2f1 = log(f2/f1);
        return arma::sin(2 * M_PI * f1 * period / logf2f1 * (arma::exp(logf2f1 * t / period) - 1));
    }

    inline void moveSet(arma::mat& set, const arma::vec& offset)
    {
        int colNb = set.n_cols;
        for (int i = 0; i < colNb; i++)
        {
            set.col(i) += offset(i);
        }
    }

    inline arma::vec getSetCentroid(const arma::mat& set)
    {
        return  arma::conv_to< arma::vec >::from(arma::mean(set, 0));
    }
}

#endif
