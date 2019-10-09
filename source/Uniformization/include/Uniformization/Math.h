#ifndef UNIFORMIZATION_MATH_H
#define UNIFORMIZATION_MATH_H

#include <armadillo>
#include <cmath>
#include <cstddef>

namespace adaptone
{
    template<class T>
    inline T logSinChirp(float f1, float f2, float period, uint32_t fs)
    {
        const std::size_t N = round(period * fs);

        T t = arma::linspace<T>(0, period, N);
        double logf2f1 = log(f2/f1);
        return arma::sin(2 * M_PI * f1 * period / logf2f1 * (arma::exp(logf2f1 * t / period) - 1));
    }

    arma::colvec linearReg(const arma::vec& y, const arma::mat& X);

    double relativePositionsFromDistances(const arma::mat& distMat, arma::mat& setAPosMat, arma::mat& setBPosMat, int iterNb,
        int tryNb, int thermalIterNb, float alpha, float epsilonTotalDistError, float epsilonDeltaTotalDistError,
        int countThreshold, int dimension);

    void rotSetAroundVec3D(arma::mat& set, arma::vec u, float angle);

    void rotSet2D(arma::mat& set, float angle);

    inline void setApplyOffset(arma::mat& set, const arma::vec& offset)
    {
        int colNb = set.n_cols;
        for (int i = 0; i < colNb; i++)
        {
            set.col(i) += offset(i);
        }
    }

    inline arma::vec setGetCentroid(const arma::mat& set)
    {
        return  arma::conv_to<arma::vec>::from(arma::mean(set, 0));
    }

    float findSetAngle2D(const arma::mat& set);
}

#endif
