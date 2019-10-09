#include "Uniformization/Math.h"

using namespace arma;
using namespace std;

colvec adaptone::linearRegression(const vec & y, const mat & X)
{
    int n = X.n_rows, k = X.n_cols;

    colvec coef = solve(X, y);
    colvec resid = y - X*coef;

    return coef;
}

double adaptone::computeRelativePositionsFromDistances(const mat& distanceMat, int iterationCount, int tryCount,
    int thermalIterationCount, double alpha, double epsilonTotalDistanceError, double epsilonDeltaTotalDistanceError,
    int countThreshold, int dimension, mat& setAPositionMat, mat& setBPositionMat)
{
    double deltaTotalDistError = INFINITY;
    double prevTotalDistError = 0;
    double totalDistanceError;
    int count = 0;
    int status = 0;

    int rowCount = distanceMat.n_rows;
    int colCount = distanceMat.n_cols;

    double avgDist = mean(mean(distanceMat));
    // initialize setA and setB position as random inside a space bound by a guessed box from distances
    if (setAPositionMat.is_empty())
    {
        setAPositionMat = avgDist * randu<mat>(rowCount, dimension);
    }

    if (setBPositionMat.is_empty())
    {
        setBPositionMat = avgDist * randu<mat>(colCount, dimension);
    }

    mat distanceNewMat = zeros(rowCount, colCount);

    for (int k = 0; k < tryCount; k++)
    {
        if (status == 1)
        {
            break;
        }

        for (int n = 0; n < iterationCount; n++)
        {
            totalDistanceError = 0;
            for (int i = 0; i < rowCount; i++)
            {
                status = 0;
                for (int j = 0; j < colCount; j++)
                {
                    mat uVec = setAPositionMat.row(i) - setBPositionMat.row(j);
                    double uNorm = norm(uVec);
                    mat uNormVec = uVec / uNorm;

                    distanceNewMat(i, j) = uNorm;

                    double distanceError = distanceNewMat(i, j) - distanceMat(i, j);
                    totalDistanceError += abs(distanceError);

                    mat distanceErrorVec = distanceError * uNormVec;
                    mat setAOffset = -alpha * 0.5 * distanceErrorVec;
                    mat setBOffset = alpha * 0.5 * distanceErrorVec;

                    double thermalNoiseFactor = 0.5 * avgDist * exp(-3.0 * (double)n / thermalIterationCount);
                    mat setAThermalOffset =
                        thermalNoiseFactor * (1 - 2 * randu<mat>(1, dimension));
                    mat setBThermalOffset =
                        thermalNoiseFactor * (1 - 2 * randu<mat>(1, dimension));

                    setAPositionMat.row(i) += setAOffset + setAThermalOffset;
                    setBPositionMat.row(j) += setBOffset + setBThermalOffset;
                }
            }

            deltaTotalDistError = abs(prevTotalDistError - totalDistanceError);
            prevTotalDistError = totalDistanceError;

            if (deltaTotalDistError < epsilonDeltaTotalDistanceError)
            {
                count++;
                if (count > countThreshold)
                {
                    if (totalDistanceError < epsilonTotalDistanceError)
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

            if (n == iterationCount && totalDistanceError < epsilonTotalDistanceError)
            {
                status = 1;
            }

            if (status == 1 || status == -1)
            {
                break;
            }
        }
    }
    return totalDistanceError;
}

void adaptone::rotateSetAroundVec3D(mat& set, vec u, double angle)
{
    u = normalise(u);

    mat rotMat = { {cos(angle) + u[0]*u[0]*(1 - cos(angle)),        u[0]*u[1]*(1 - cos(angle)) - u[2]*sin(angle),   u[0]*u[2]*(1 - cos(angle)) + u[1]*sin(angle)},
                   {u[1]*u[0]*(1 - cos(angle)) + u[2]*sin(angle),   cos(angle) + u[1]*u[1]*(1 - cos(angle)),        u[1]*u[2]*(1 - cos(angle)) - u[0]*sin(angle)},
                   {u[2]*u[0]*(1 - cos(angle)) - u[1]*sin(angle),   u[2]*u[1]*(1 - cos(angle)) + u[0]*sin(angle),   cos(angle) + u[2]*u[2]*(1 - cos(angle))} };

    int rowCount = set.n_rows;
    for (int i = 0; i < rowCount; i++)
    {
        set.cols(0,2).row(i) = trans(rotMat * set.cols(0,2).row(i).t());
    }
}

void adaptone::rotateSet2D(mat& set, double angle)
{
    mat rotMat = { {cos(angle), -sin(angle)},
                   {sin(angle),  cos(angle)} };

    int rowCount = set.n_rows;
    for (int i = 0; i < rowCount; i++)
    {
        set.cols(0,1).row(i) = trans(rotMat * set.cols(0,1).row(i).t());
    }
}

double adaptone::findSetAngle2D(const mat& set)
{
    vec y = set.col(1);

    mat X = ones(set.n_rows, 2);
    X.col(1) = set.col(0);

    vec coeff = linearRegression(y, X);

    return atan2(coeff(1),1);
}
