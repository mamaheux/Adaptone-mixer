#ifndef UNIFORMIZATION_AUTOPOSITION_H
#define UNIFORMIZATION_AUTOPOSITION_H

#include <armadillo>
#include <cmath>

namespace adaptone
{
    bool autoPosition(arma::mat distMat, int iterNb, int thermalIterNb, float alpha)
    {
        double deltaTotalDistError = INFINITY;
        double prevTotalDistError = 0;
        int count = 0;

        int rowNb = distMat.n_rows;
        int colNb = distMat.n_cols;

        double spaceFactor = arma::mean(arma::mean(distMat)) / (rowNb * colNb);

        // inirialize speaker and mic position as random inside a space bound by a guessed box from distances
        arma::mat setAPosMat = spaceFactor * arma::randu<arma::mat>(rowNb, 3);
        arma::mat setBPosMat = spaceFactor * arma::randu<arma::mat>(colNb, 3);
        arma::mat distNewMat = arma::zeros(rowNb, colNb);
        for(int n = 0; n < iterNb; n++)
        {
            double totalDistError = 0;
            for(int i = 0; i < rowNb; i++)
            {
                int status = 0;
                for(int j = 0; j < colNb; j++)
                {
                    arma::vec uVec = setAPosMat.row(i) - setBPosMat.row(j);
                    double uNorm = arma::norm(uVec);
                    arma::vec uNormVec = uVec / uNorm;

                    distNewMat(i,j) = uNorm;

                    double distError = distNewMat(i,j) - distMat(i,j);
                    totalDistError += distError;

                    arma::vec distErrorVec = distError * uNormVec;
                    arma::vec setAOffset = -1 / rowNb * alpha * 0.5 * distErrorVec;
                    arma::vec setBOffset = 1 / rowNb * alpha * 0.5 * distErrorVec;

                    double thermalNoiseFactor = std::fmax(0.0, 0.2 * (thermalIterNb - n) / thermalIterNb);

                }
            }
        }
    }
}

#endif
