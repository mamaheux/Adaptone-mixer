#include <Uniformization/Math.h>


using namespace arma;
using namespace std;

namespace adaptone
{

    arma::colvec linearReg(const arma::vec & y, const arma::mat & X) {

        int n = X.n_rows, k = X.n_cols;

        arma::colvec coef = arma::solve(X, y);
        arma::colvec resid = y - X*coef;

        double sig2 = arma::as_scalar(arma::trans(resid)*resid/(n-k));

        return coef;
    }

    double relativePositionsFromDistances(const mat& distMat, mat& setAPosMat, mat& setBPosMat, int iterNb, int tryNb,
        int thermalIterNb, float alpha, float epsilonTotalDistError, float epsilonDeltaTotalDistError,
        int countThreshold, int dimension)
    {
        double deltaTotalDistError = INFINITY;
        double prevTotalDistError = 0;
        double totalDistError;
        int count = 0;
        int status = 0;

        int rowNb = distMat.n_rows;
        int colNb = distMat.n_cols;

        double avgDist = mean(mean(distMat));
        // initialize setA and setB position as random inside a space bound by a guessed box from distances
        if (setAPosMat.is_empty())
        {
            setAPosMat = avgDist * randu<mat>(rowNb, dimension);
        }

        if (setBPosMat.is_empty())
        {
            setBPosMat = avgDist * randu<mat>(colNb, dimension);
        }

        mat distNewMat = zeros(rowNb, colNb);

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
                        mat uVec = setAPosMat.row(i) - setBPosMat.row(j);
                        double uNorm = norm(uVec);
                        mat uNormVec = uVec / uNorm;

                        distNewMat(i, j) = uNorm;

                        double distError = distNewMat(i, j) - distMat(i, j);
                        totalDistError += abs(distError);

                        mat distErrorVec = distError * uNormVec;
                        mat setAOffset = -alpha * 0.5 * distErrorVec;
                        mat setBOffset = alpha * 0.5 * distErrorVec;

                        //double thermalNoiseFactor = fmax(0.0, 0.2 * (thermalIterNb - n) / thermalIterNb);
                        double thermalNoiseFactor = 0.5 * avgDist * exp(-5 * n / thermalIterNb);
                        mat setAThermalOffset =
                            thermalNoiseFactor * (1 - 2 * randu<mat>(1, dimension));
                        mat setBThermalOffset =
                            thermalNoiseFactor * (1 - 2 * randu<mat>(1, dimension));

                        setAPosMat.row(i) += setAOffset + setAThermalOffset;
                        setBPosMat.row(j) += setBOffset + setBThermalOffset;
                    }
                }

                deltaTotalDistError = abs(prevTotalDistError - totalDistError);
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

    void rotSetAroundVec3D(mat& set, vec u, float angle)
    {
        //normalize u if not already
        u = normalise(u);

        mat rotMat = { {cos(angle) + u[0]*u[0]*(1 - cos(angle)),        u[0]*u[1]*(1 - cos(angle)) - u[2]*sin(angle),   u[0]*u[2]*(1 - cos(angle)) + u[1]*sin(angle)},
                       {u[1]*u[0]*(1 - cos(angle)) + u[2]*sin(angle),   cos(angle) + u[1]*u[1]*(1 - cos(angle)),        u[1]*u[2]*(1 - cos(angle)) - u[0]*sin(angle)},
                       {u[2]*u[0]*(1 - cos(angle)) - u[1]*sin(angle),   u[2]*u[1]*(1 - cos(angle)) + u[0]*sin(angle),   cos(angle) + u[2]*u[2]*(1 - cos(angle))} };

        int rowNb = set.n_rows;
        for (int i = 0; i < rowNb; i++)
        {
            set.cols(0,2).row(i) = trans(rotMat * set.cols(0,2).row(i).t());
        }
    }

    void rotSet2D(mat& set, float angle)
    {
        mat rotMat = { {cos(angle), -sin(angle)},
                       {sin(angle),  cos(angle)} };

        int rowNb = set.n_rows;
        for (int i = 0; i < rowNb; i++)
        {
            set.cols(0,1).row(i) = trans(rotMat * set.cols(0,1).row(i).t());
        }
    }

    float findSetAngle2D(const arma::mat& set)
    {
        vec y = set.col(1);

        mat X = ones(set.n_rows, 2);
        X.col(1) = set.col(0);

        vec coeff = linearReg(y, X);

        return atan2(coeff(1),1);
    }
}
