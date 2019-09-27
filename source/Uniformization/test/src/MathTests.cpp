#include <Uniformization/Math.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(MathTests, logSinChirp_shouldReturnTheLogSinChirp)
{
    constexpr float T = 1;
    constexpr uint32_t Fs = 44100;
    constexpr float Tol = 0.001;

    arma::fvec chirp = logSinChirp<arma::fvec>(20.0, 10000.0, T, Fs);

    EXPECT_EQ(chirp.n_elem, 44100);
    EXPECT_NEAR(chirp(0), 0.0, Tol);
    EXPECT_NEAR(chirp(200), 0.54637, Tol);
    EXPECT_NEAR(chirp(5001), 0.96303, Tol);
    EXPECT_NEAR(chirp(20000), -0.93309, Tol);
    EXPECT_NEAR(chirp(44099), -0.61934, Tol);
}

TEST(MathTests, relativePositionFromDistance_2D_shouldGetRelativePositionFromDistance)
{
    constexpr int TestNb = 1000;

    constexpr double Alpha = 1.0;
    constexpr double EspilonTotalDistError = 1e-3;
    constexpr double EspilonDeltaTotalDistError = 1e-5;

    constexpr int IterNb = 500;
    constexpr int ThermalIterNb = 50;
    constexpr int TryNb = 5;
    constexpr int CountThreshold = 5;
    constexpr int dimension = 2;

    int passedNb = 0;
    for(int n = 0; n < TestNb; n++)
    {
        int setANb = std::rand() % 29 + 4; //value in range [4, 32]
        int setBNb = std::rand() % 29 + 4; //value in range [4, 32]

        arma::mat setAPosMat = 10 * arma::randu<arma::mat>(setANb, dimension);
        arma::mat setBPosMat = 10 * arma::randu<arma::mat>(setBNb, dimension);
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

        double totalDistError = relativePositionsFromDistances(distMat, setAPosNewMat, setBPosNewMat, IterNb, TryNb,
            ThermalIterNb, Alpha, EspilonTotalDistError, EspilonDeltaTotalDistError, CountThreshold, dimension);

        if (totalDistError < EspilonTotalDistError)
        {
            passedNb++;
        }
    }

    EXPECT_TRUE( passedNb / TestNb >= 0.99);
}

TEST(MathTests, relativePositionFromDistance_3D_shouldGetRelativePositionFromDistance)
{
    constexpr int TestNb = 1000;

    constexpr double Alpha = 1.0;
    constexpr double EspilonTotalDistError = 1e-3;
    constexpr double EspilonDeltaTotalDistError = 1e-5;

    constexpr int IterNb = 500;
    constexpr int ThermalIterNb = 50;
    constexpr int TryNb = 5;
    constexpr int CountThreshold = 5;
    constexpr int dimension = 3;

    int passedNb = 0;
    for(int n = 0; n < TestNb; n++)
    {
        int setANb = std::rand() % 29 + 4; //value in range [4, 32]
        int setBNb = std::rand() % 29 + 4; //value in range [4, 32]

        arma::mat setAPosMat = 10 * arma::randu<arma::mat>(setANb, dimension);
        arma::mat setBPosMat = 10 * arma::randu<arma::mat>(setBNb, dimension);
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

        double totalDistError = relativePositionsFromDistances(distMat, setAPosNewMat, setBPosNewMat, IterNb, TryNb,
            ThermalIterNb, Alpha, EspilonTotalDistError, EspilonDeltaTotalDistError, CountThreshold, dimension);

        if (totalDistError < EspilonTotalDistError)
        {
            passedNb++;
        }
    }

    EXPECT_TRUE( passedNb / TestNb >= 0.99);
}
