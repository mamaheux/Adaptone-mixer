#include <Uniformization/autoPosition.h>

#include <armadillo>
#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(autoPositionTests, autoPosition3DNoNoise_shouldGetRelativePositionFromDistance)
{
    constexpr int TestNb = 1000;

    constexpr double Alpha = 1.0;
    constexpr double EspilonTotalDistError = 1e-3;
    constexpr double EspilonDeltaTotalDistError = 1e-5;

    constexpr int IterNb = 500;
    constexpr int ThermalIterNb = 50;
    constexpr int TryNb = 5;
    constexpr int CountThreshold = 5;

    int passedNb = 0;
    for(int n = 0; n < TestNb; n++)
    {
        int setANb = std::rand() % 29 + 4; //value in range [4, 32]
        int setBNb = std::rand() % 29 + 4; //value in range [4, 32]

        arma::mat setAPosMat = 10 * arma::randu<arma::mat>(setANb, 3);
        arma::mat setBPosMat = 10 * arma::randu<arma::mat>(setBNb, 3);
        arma::mat distMat = arma::zeros<arma::mat>(setANb, setBNb);

        // compute distance between each pair of the two set (A and B)
        for (int i = 0; i < setANb; i++)
        {
            for (int j = 0; j < setBNb; j++)
            {
                distMat(i, j) = arma::norm(setAPosMat.row(i) - setBPosMat.row(j));
            }
        }

        arma::mat setAPosNewMat;
        arma::mat setBPosNewMat;
        double avgDist = arma::mean(arma::mean(distMat));
        //double epsilonTotalDistError = avgDist * distRelativeError;

        double totalDistError = autoPosition(distMat, setAPosNewMat, setBPosNewMat, IterNb, TryNb, ThermalIterNb,
            Alpha, EspilonTotalDistError, EspilonDeltaTotalDistError, CountThreshold);

        if (totalDistError < EspilonTotalDistError)
        {
            passedNb++;
        }
    }

    EXPECT_TRUE( passedNb / TestNb >= 0.99);
}
