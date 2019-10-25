#include <UniformizationTests/ArmadilloUtils.h>

#include <Uniformization/Math.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;
using namespace arma;

TEST(MathTests, logSinChirp_shouldReturnTheLogSinChirp)
{
    constexpr float Tolerance = 0.001;
    constexpr float Period = 1;
    constexpr size_t Fs = 44100;

    fvec chirp = logSinChirp<fvec>(20.0, 10000.0, Period, Fs);

    EXPECT_EQ(chirp.n_elem, 44100);
    EXPECT_NEAR(chirp(0), 0.0, Tolerance);
    EXPECT_NEAR(chirp(200), 0.54637, Tolerance);
    EXPECT_NEAR(chirp(5001), 0.96303, Tolerance);
    EXPECT_NEAR(chirp(20000), -0.93309, Tolerance);
    EXPECT_NEAR(chirp(44099), -0.61934, Tolerance);
}

TEST(MathTests, linearRegression_shouldReturnTheGoodCoefficients)
{
    constexpr float Tolerance = 0.00001;

    const mat Set =
    {
        { 0, 1 },
        { 1, 2 },
        { -1, -2 },
        { 4, 5 },
        { 0, -1 },
        { 0.5, 0.333 },
        { 0, 0 }
    };

    mat X = ones(arma::size(Set));
    X.col(1) = Set.col(0);

    vec coeff = linearRegression(Set.col(1), X);

    EXPECT_NEAR(coeff(0), -0.10625, Tolerance);
    EXPECT_NEAR(coeff(1), 1.35039, Tolerance);
}

TEST(MathTests, computeRelativePositionsFromDistances_2D_shouldGetRelativePositionFromDistance)
{
    constexpr int TestCount = 25;

    constexpr double Alpha = 1.0;
    constexpr double EpsilonTotalDistanceError = 5e-3;
    constexpr double EpsilonDeltaTotalDistError = 1e-6;

    constexpr size_t IterationCount = 750;
    constexpr size_t ThermalIterationCount = 100;
    constexpr size_t TryCount = 5;
    constexpr size_t CountThreshold = 5;
    constexpr size_t Dimension = 2;

    int passedCount = 0;
    for(int n = 0; n < TestCount; n++)
    {
        size_t setACount = rand() % 13 + 4; //value in range [4, 16]
        size_t setBCount = rand() % 13 + 4; //value in range [4, 16]

        mat setAPosMat = 10 * randu<mat>(setACount, Dimension);
        mat setBPosMat = 10 * randu<mat>(setBCount, Dimension);
        mat distMat = zeros<mat>(setACount, setBCount);

        // compute distance between each pair of the two set (A and B)
        for (int i = 0; i < setACount; i++)
        {
            for (int j = 0; j < setBCount; j++)
            {
                distMat(i, j) = norm(setAPosMat.row(i) - setBPosMat.row(j));
            }
        }

        mat setAPositionNewMat;
        mat setBPositionNewMat;

        double totalDistanceError = computeRelativePositionsFromDistances(distMat, IterationCount, TryCount,
            ThermalIterationCount, Alpha, EpsilonTotalDistanceError, EpsilonDeltaTotalDistError, CountThreshold,
            Dimension, setAPositionNewMat, setBPositionNewMat);

        if (totalDistanceError < EpsilonTotalDistanceError)
        {
            passedCount++;
        }
    }

    EXPECT_TRUE(passedCount >= TestCount - 2);
}

TEST(MathTests, computeRelativePositionsFromDistances_3D_shouldGetRelativePositionFromDistance)
{
    constexpr int TestCount = 25;

    constexpr double Alpha = 1.0;
    constexpr double EpsilonTotalDistanceError = 5e-3;
    constexpr double EpsilonDeltaTotalDistError = 1e-6;

    constexpr size_t IterationCount = 2000;
    constexpr size_t ThermalIterationCount = 50;
    constexpr size_t TryCount = 10;
    constexpr size_t CountThreshold = 5;
    constexpr size_t Dimension = 3;

    int passedCount = 0;
    for(int n = 0; n < TestCount; n++)
    {
        size_t setACount = rand() % 13 + 4; //value in range [4, 16]
        size_t setBCount = rand() % 13 + 4; //value in range [4, 16]

        mat setAPosMat = 10 * randu<mat>(setACount, Dimension);
        mat setBPosMat = 10 * randu<mat>(setBCount, Dimension);
        mat distMat = zeros<mat>(setACount, setBCount);

        // compute distance between each pair of the two set (A and B)
        for (int i = 0; i < setACount; i++)
        {
            for (int j = 0; j < setBCount; j++)
            {
                distMat(i, j) = norm(setAPosMat.row(i) - setBPosMat.row(j));
            }
        }

        mat setAPositionNewMat;
        mat setBPositionNewMat;

        double totalDistanceError = computeRelativePositionsFromDistances(distMat, IterationCount, TryCount,
            ThermalIterationCount, Alpha, EpsilonTotalDistanceError, EpsilonDeltaTotalDistError, CountThreshold,
            Dimension, setAPositionNewMat, setBPositionNewMat);

        if (totalDistanceError < EpsilonTotalDistanceError)
        {
            passedCount++;
        }
    }

    EXPECT_TRUE(passedCount >= TestCount - 2);
}

TEST(MathTests, rotateSet2D_shouldApplyProperRotationToAllPointsInTheSet)
{
    constexpr float Tolerance = 0.00001;

    mat set =
    {
        { 0, 1 },
        { 1, 2 },
        { -1, -2 },
        { 4, 5 },
        { 0, -1 },
        { 0.5, 0.333 },
        { 0, 0 }
    };

    rotateSet2D(set, 0.25);

    const mat SetTarget =
    {
        { -0.24740, 0.96891 },
        { 0.47410, 2.18523 },
        { -0.47410, -2.18523 },
        { 2.63863, 5.83418 },
        { 0.24740, -0.96891 },
        { 0.40207, 0.44635 },
        { 0.00000, 0.00000 }
    };

    EXPECT_MAT_NEAR(set, SetTarget, Tolerance);
}

TEST(MathTests, rotateSetAroundVec3D_shouldApplyProperRotationToAllPointsInTheSet)
{
    constexpr float Tolerance = 0.00001;

    mat set =
    {
        { 0, 1, 2 },
        { 1, 2, 3 },
        { -1, -2, -3 },
        { 4, 5, 9 },
        { 0, -1, -7 },
        { 0.5, 0.333, 1.4 },
        { 0, 0, 0 }
    };
    vec unitVec = {0,0,1};

    rotateSetAroundVec3D(set, unitVec, 0.25);

    mat setTarget =
    {
        { -0.24740, 0.96891, 2.00000 },
        { 0.47410, 2.18523, 3.00000 },
        { -0.47410, -2.18523, -3.00000 },
        { 2.63863, 5.83418, 9.00000 },
        { 0.24740, -0.96891, -7.00000 },
        { 0.40207, 0.44635, 1.40000 },
        { 0.00000, 0.00000, 0.00000 }
    };

    EXPECT_MAT_NEAR(set, setTarget, Tolerance);

    vec vec = {-1.7, 2.45, 1.2};

    rotateSetAroundVec3D(set, vec, 2.64);

    setTarget =
    {
        { -0.82803, 1.92625, -0.77712 },
        { -2.22450, 2.55790, -1.58390 },
        { 2.22450 , -2.55790, 1.58390 },
        { -6.42613, 6.81781, -5.84999 },
        { 0.84839 , -5.86785, 3.85339 },
        { -0.56498, 0.96693, -1.03283 },
        { 0.00000 , 0.00000, 0.00000 }
    };

    EXPECT_MAT_NEAR(set, setTarget, Tolerance);
}

TEST(MathTests, moveSet_shouldApplyOffsetToAllPointsOfASet)
{
    constexpr float Tolerance = 0.00001;

    mat set =
    {
        { 0, 1 },
        { 1, 2 },
        { -1, -2 },
    };

    vec offset = {1, -2.5};

    moveSet(set, offset);

    const mat SetTarget =
    {
        { 1, -1.5 },
        { 2, -0.5 },
        { 0, -4.5 },
    };

    EXPECT_MAT_NEAR(set, SetTarget, Tolerance);
}

TEST(MathTests, getSetCentroid_shouldGetTheCentroidOfASet)
{
    constexpr float Tolerance = 0.00001;

    mat set =
    {
        { 0,  1 },
        { 1,  2 },
        { -1, -2 },
    };

    vec centroid = getSetCentroid(set);

    EXPECT_NEAR(centroid(0), 0, Tolerance);
    EXPECT_NEAR(centroid(1), 0.33333333333, Tolerance);
}

TEST(MathTests, findSetAngle2D_shouldReturnAngleFromXAxisAndSetOrientation)
{
    constexpr float Tolerance = 0.00001;

    mat set =
    {
        { 0, 1 },
        { 1, 2 },
        { -1, -2 },
        { 4, 5 },
        { 0, -1 },
        { 0.5, 0.333 },
        { 0, 0 }
    };

    double angle = findSetAngle2D(set);

    EXPECT_NEAR(angle, 0.93339, Tolerance);
}

TEST(MathTests, correlation_shouldReturnTheCorrelationVectorBetweenTwoVector)
{
    constexpr double Tolerance = 0.0001;
    const vec A = { -1, 2, 3, 4, 5, 6, 9, -2, -3, 0, 4 };
    const vec B = { 5, 4, 9, -3, 2, 0, -5, 2, -6, 4, 5 };

    const vec RTarget = { -5, 6, 29, 18, 32, 22, 30, -13, -99, 1, 17, 75, 67, 71, 110, 57, 18, -34, 21, 16, 20 };
    vec R = correlation(A, B);

    EXPECT_VEC_NEAR(R, RTarget, Tolerance);
}

TEST(MathTests, findDelay_shouldReturnTheCorrelationVectorBetweenTwoVector)
{
    const vec A = { 0, 0, 0, 1, 2, 3, 2, 1, -1, -2, -3, -2, -1, 0, 0, 0 };
    const vec B = { 0, 0, 0, 0, 0, 0, 1, 2, 3, 2, 1, -1, -2, -3, -2, -1 };

    size_t delayAB = findDelay(A, B);
    size_t delayBA = findDelay(B, A);

    EXPECT_EQ(delayAB, -3);
    EXPECT_EQ(delayBA, 3);
}

TEST(MathTests, averageFrequencyBand_shouldReturnVectorContainingTheAverageFrequencyBandAmplitudeInDB)
{
    constexpr double Tolerance = 0.0001;
    constexpr size_t Fs = 44100;
    const vec Fc = { 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
                      2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000 };
    const vec Weights = linspace(0.5, 1, Fc.size());

    const vec bandAverageTarget = { 25.1680, 34.8295, 42.0367, 29.8513, 21.5859, 15.3491, 10.2959, 5.7486, 1.7702,
                                    -1.7061, -4.2969, -6.4862, -8.2789, -9.1462, -9.8197, -10.6223, -10.3798, -10.1308,
                                    -10.8167, -10.5026, -9.4813, -9.5543, -9.8470, -9.4832, -8.9783, -9.1044, -8.9512,
                                    -8.2119, -7.8026, -7.6590, -5.3756 };
    // Construct a signal with frequency of each band and with increasing weight
    vec X = zeros<vec>(Fs);
    for (int i = 0; i < Fc.size(); i++)
    {
        X += Weights(i) * sin(2 * M_PI * Fc(i) * linspace(0, 1, Fs));
    }

    arma::vec bandAverage = averageFrequencyBand(X, Fc, Fs);

    EXPECT_VEC_NEAR(bandAverage, bandAverageTarget, Tolerance);
}


