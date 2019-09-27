#include "Uniformization/Math.h"

namespace adaptone
{
    double relativePositionsFromDistances(arma::mat distMat, arma::mat setAPosMat, arma::mat setBPosMat, int iterNb, int tryNb,
        int thermalIterNb, float alpha, float epsilonTotalDistError, float epsilonDeltaTotalDistError,
        int countThreshold, int dimension)
    {
        double deltaTotalDistError = INFINITY;
        double prevTotalDistError = 0;
        double totalDistError;
        int count = 0;
        int status;

        int rowNb = distMat.n_rows;
        int colNb = distMat.n_cols;

        double avgDist = arma::mean(arma::mean(distMat));
        // initialize speaker and mic position as random inside a space bound by a guessed box from distances
        if (setAPosMat.is_empty())
        {
            setAPosMat = avgDist * arma::randu<arma::mat>(rowNb, dimension);
        }

        if (setBPosMat.is_empty())
        {
            setBPosMat = avgDist * arma::randu<arma::mat>(colNb, dimension);
        }

        arma::mat distNewMat = arma::zeros(rowNb, colNb);

        for (int k = 0; k < tryNb; k++)
        {
            if (status == 1)
            {
                break;
            }

            for (int n = 0; n < iterNb; n++)
            {
                totalDistError = 0;
                for (int i = 0; i < rowNb; i++)
                {
                    status = 0;
                    for (int j = 0; j < colNb; j++)
                    {
                        arma::mat uVec = setAPosMat.row(i) - setBPosMat.row(j);
                        double uNorm = arma::norm(uVec);
                        arma::mat uNormVec = uVec / uNorm;

                        distNewMat(i, j) = uNorm;

                        double distError = distNewMat(i, j) - distMat(i, j);
                        totalDistError += std::abs(distError);

                        arma::mat distErrorVec = distError * uNormVec;
                        arma::mat setAOffset = -alpha * 0.5 * distErrorVec;
                        arma::mat setBOffset = alpha * 0.5 * distErrorVec;

                        //double thermalNoiseFactor = std::fmax(0.0, 0.2 * (thermalIterNb - n) / thermalIterNb);
                        double thermalNoiseFactor = 0.5 * avgDist * std::exp(-5 * n / thermalIterNb);
                        arma::mat setAThermalOffset =
                            thermalNoiseFactor * (1 - 2 * arma::randu<arma::mat>(1, dimension));
                        arma::mat setBThermalOffset =
                            thermalNoiseFactor * (1 - 2 * arma::randu<arma::mat>(1, dimension));

                        setAPosMat.row(i) += setAOffset + setAThermalOffset;
                        setBPosMat.row(j) += setBOffset + setBThermalOffset;
                    }
                }

                deltaTotalDistError = std::abs(prevTotalDistError - totalDistError);
                prevTotalDistError = totalDistError;

                if (deltaTotalDistError < epsilonDeltaTotalDistError)
                {
                    count++;
                    if (count > countThreshold)
                    {
                        if (totalDistError < epsilonTotalDistError)
                        {
                            status = 1;
                            break;
                        }
                        else
                        {
                            status = -1;
                            break;
                        }
                    }
                }

                if (n == iterNb && totalDistError < epsilonTotalDistError)
                {
                    status = 1;
                }

                if (status == 1 || status == -1)
                {
                    break;
                }
            }
        }
        return totalDistError;
    }
}