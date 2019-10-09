#include <Uniformization/Math.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(MathTests, logSinChirp_shouldReturnTheLogSinChirp)
{
    constexpr float Tolerance = 0.001;
    constexpr float Period = 1;
    constexpr uint32_t Fs = 44100;

    arma::fvec chirp = logSinChirp<arma::fvec>(20.0, 10000.0, Period, Fs);

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

    const arma::mat set = {
        { 0, 1 },
        { 1, 2 },
        { -1, -2 },
        { 4, 5 },
        { 0, -1 },
        { 0.5, 0.333 },
        { 0, 0 }
    };

    arma::mat X = arma::ones(arma::size(set));
    X.col(1) = set.col(0);

    arma::vec coeff = linearRegression(set.col(1), X);

    EXPECT_NEAR(coeff(0),-0.10625, Tolerance);
    EXPECT_NEAR(coeff(1),1.35039, Tolerance);
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

        arma::mat setAPosMat = 10 * arma::randu<arma::mat>(setACount, Dimension);
        arma::mat setBPosMat = 10 * arma::randu<arma::mat>(setBCount, Dimension);
        arma::mat distMat = arma::zeros<arma::mat>(setACount, setBCount);

        // compute distance between each pair of the two set (A and B)
        for (int i = 0; i < setACount; i++)
        {
            for (int j = 0; j < setBCount; j++)
            {
                distMat(i, j) = arma::norm(setAPosMat.row(i) - setBPosMat.row(j));
            }
        }

        arma::mat setAPositionNewMat;
        arma::mat setBPositionNewMat;

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
    constexpr size_t ThermalIterationCount = 100;
    constexpr size_t TryCount = 5;
    constexpr size_t CountThreshold = 5;
    constexpr size_t Dimension = 3;

    int passedCount = 0;
    for(int n = 0; n < TestCount; n++)
    {
        size_t setACount = std::rand() % 13 + 4; //value in range [4, 16]
        size_t setBCount = std::rand() % 13 + 4; //value in range [4, 16]

        arma::mat setAPosMat = 10 * arma::randu<arma::mat>(setACount, Dimension);
        arma::mat setBPosMat = 10 * arma::randu<arma::mat>(setBCount, Dimension);
        arma::mat distMat = arma::zeros<arma::mat>(setACount, setBCount);

        // compute distance between each pair of the two set (A and B)
        for (int i = 0; i < setACount; i++)
        {
            for (int j = 0; j < setBCount; j++)
            {
                distMat(i, j) = arma::norm(setAPosMat.row(i) - setBPosMat.row(j));
            }
        }

        arma::mat setAPositionNewMat;
        arma::mat setBPositionNewMat;

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

    arma::mat set = {
        { 0, 1 },
        { 1, 2 },
        { -1, -2 },
        { 4, 5 },
        { 0, -1 },
        { 0.5, 0.333 },
        { 0, 0 }
    };

    rotateSet2D(set, 0.25);

    const arma::mat SetTarget = {
        { -0.24740, 0.96891 },
        { 0.47410, 2.18523 },
        { -0.47410, -2.18523 },
        { 2.63863, 5.83418 },
        { 0.24740, -0.96891 },
        { 0.40207, 0.44635 },
        { 0.00000, 0.00000 }
    };

    // test every matrix entries - successive ROT 1
    int rowCount = set.n_rows;
    int colCount = set.n_cols;
    for (int i = 0; i < rowCount; i++)
    {
        for (int j = 0; j < colCount; j++)
        {
            EXPECT_NEAR(set(i,j), SetTarget(i,j), Tolerance);
        }
    }
}

TEST(MathTests, rotateSetAroundVec3D_shouldApplyProperRotationToAllPointsInTheSet)
{
    constexpr float Tolerance = 0.00001;

    arma::mat set = {
        { 0, 1, 2 },
        { 1, 2, 3 },
        { -1, -2, -3 },
        { 4, 5, 9 },
        { 0, -1, -7 },
        { 0.5, 0.333, 1.4 },
        { 0, 0, 0 }
    };
    arma::vec unitVec = {0,0,1};

    rotateSetAroundVec3D(set, unitVec, 0.25);

    arma::mat setTarget = {
        { -0.24740, 0.96891, 2.00000 },
        { 0.47410, 2.18523, 3.00000 },
        { -0.47410, -2.18523, -3.00000 },
        { 2.63863, 5.83418, 9.00000 },
        { 0.24740, -0.96891, -7.00000 },
        { 0.40207, 0.44635, 1.40000 },
        { 0.00000, 0.00000, 0.00000 }
    };

    // test every matrix entries - successive ROT 1
    int rowCount = set.n_rows;
    int colCount = set.n_cols;
    for (int i = 0; i < rowCount; i++)
    {
        for (int j = 0; j < colCount; j++)
        {
            EXPECT_NEAR(set(i,j), setTarget(i,j), Tolerance);
        }
    }

    arma::vec vec = {-1.7, 2.45, 1.2};

    rotateSetAroundVec3D(set, vec, 2.64);

    setTarget = {
        { -0.82803, 1.92625, -0.77712 },
        { -2.22450, 2.55790, -1.58390 },
        { 2.22450 , -2.55790, 1.58390 },
        { -6.42613, 6.81781, -5.84999 },
        { 0.84839 , -5.86785, 3.85339 },
        { -0.56498, 0.96693, -1.03283 },
        { 0.00000 , 0.00000, 0.00000 }
    };

    // test every matrix entries - successive ROT 2
    for (int i = 0; i < rowCount; i++)
    {
        for (int j = 0; j < colCount; j++)
        {
            EXPECT_NEAR(set(i,j), setTarget(i,j), Tolerance);
        }
    }
}

TEST(MathTests, moveSet_shouldApplyOffsetToAllPointsOfASet)
{
    constexpr float Tolerance = 0.00001;

    arma::mat set = {
        { 0, 1 },
        { 1, 2 },
        { -1, -2 },
    };

    arma::vec offset = {1, -2.5};

    moveSet(set, offset);

    const arma::mat SetTarget = {
        { 1, -1.5 },
        { 2, -0.5 },
        { 0, -4.5 },
    };

    // test every matrix entries
    for (int i = 0; i < set.n_rows; i++)
    {
        for (int j = 0; j < set.n_cols; j++)
        {
            EXPECT_NEAR(set(i,j), SetTarget(i,j), Tolerance);
        }
    }
}

TEST(MathTests, getSetCentroid_shouldGetTheCentroidOfASet)
{
    constexpr float Tolerance = 0.00001;

    arma::mat set = {
        { 0,  1 },
        { 1,  2 },
        { -1, -2 },
    };

    arma::vec centroid = getSetCentroid(set);

    EXPECT_NEAR(centroid(0), 0, Tolerance);
    EXPECT_NEAR(centroid(1), 0.33333333333, Tolerance);
}

TEST(MathTests, findSetAngle2D_shouldReturnAngleFromXAxisAndSetOrientation)
{
    constexpr float Tolerance = 0.00001;

    arma::mat set = {
        { 0, 1 },
        { 1, 2 },
        { -1, -2 },
        { 4, 5 },
        { 0, -1 },
        { 0.5, 0.333 },
        { 0, 0 }
    };

    float angle = findSetAngle2D(set);

    EXPECT_NEAR(angle, 0.93339, Tolerance);
}
