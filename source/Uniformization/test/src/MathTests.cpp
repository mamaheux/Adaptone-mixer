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

TEST(MathTests, linearReg_shouldReturnTheGoodCoefficients)
{
    arma::mat set = {
        {0, 1},
        {1, 2},
        {-1, -2},
        {4, 5},
        {0, -1},
        {0.5, 0.333},
        {0, 0}
    };

    arma::mat X = arma::ones(arma::size(set));
    X.col(1) = set.col(0);

    arma::vec coeff = linearReg(set.col(1), X);

    constexpr float Tol = 0.00001;
    EXPECT_NEAR(coeff(0),-0.10625, Tol);
    EXPECT_NEAR(coeff(1),1.35039, Tol);
}

TEST(MathTests, relativePositionFromDistance_2D_shouldGetRelativePositionFromDistance)
{
    constexpr int TestNb = 25;

    constexpr double Alpha = 1.0;
    constexpr double EspilonTotalDistError = 5e-3;
    constexpr double EspilonDeltaTotalDistError = 1e-6;

    constexpr int IterNb = 750;
    constexpr int ThermalIterNb = 100;
    constexpr int TryNb = 5;
    constexpr int CountThreshold = 5;
    constexpr int dimension = 2;

    int passedNb = 0;
    for(int n = 0; n < TestNb; n++)
    {
        int setANb = std::rand() % 13 + 4; //value in range [4, 16]
        int setBNb = std::rand() % 13 + 4; //value in range [4, 16]

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

        //double avgDist = arma::mean(arma::mean(distMat));
        //double epsilonTotalDistError = avgDist * distRelativeError;

        double totalDistError = relativePositionsFromDistances(distMat, setAPosNewMat, setBPosNewMat, IterNb, TryNb,
            ThermalIterNb, Alpha, EspilonTotalDistError, EspilonDeltaTotalDistError, CountThreshold, dimension);

        if (totalDistError < EspilonTotalDistError)
        {
            passedNb++;
        }
    }

    EXPECT_TRUE( passedNb / TestNb >= (TestNb-2)/TestNb);
}

TEST(MathTests, relativePositionFromDistance_3D_shouldGetRelativePositionFromDistance)
{
    constexpr int TestNb = 25;

    constexpr double Alpha = 1.0;
    constexpr double EspilonTotalDistError = 5e-3;
    constexpr double EspilonDeltaTotalDistError = 1e-6;

    constexpr int IterNb = 750;
    constexpr int ThermalIterNb = 100;
    constexpr int TryNb = 5;
    constexpr int CountThreshold = 5;
    constexpr int dimension = 3;

    int passedNb = 0;
    for(int n = 0; n < TestNb; n++)
    {
        int setANb = std::rand() % 13 + 4; //value in range [4, 16]
        int setBNb = std::rand() % 13 + 4; //value in range [4, 16]

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

    EXPECT_TRUE( passedNb / TestNb >= (TestNb-2)/TestNb);
}

TEST(MathTests, rotSet2D_shouldApplyProperRotationToAllPointsInTheSet)
{
    arma::mat set = {
        {0, 1},
        {1, 2},
        {-1, -2},
        {4, 5},
        {0, -1},
        {0.5, 0.333},
        {0, 0}
    };

    rotSet2D(set, 0.25);

    constexpr float Tol = 0.00001;
    arma::mat setTarget = {
        {-0.24740,   0.96891},
        {0.47410,   2.18523},
        {-0.47410,  -2.18523},
        {2.63863,   5.83418},
        {0.24740,  -0.96891},
        {0.40207,   0.44635},
        {0.00000,   0.00000}
    };

    // test every matrix entries - successive ROT 1
    int rowNb = set.n_rows;
    int colNb = set.n_cols;
    for (int i = 0; i < rowNb; i++)
    {
        for (int j = 0; j < colNb; j++)
        {
            EXPECT_NEAR(set(i,j), setTarget(i,j), Tol);
        }
    }


}

TEST(MathTests, rotSetAroundVec3D_shouldApplyProperRotationToAllPointsInTheSet)
{
    arma::mat set = {
        {0, 1, 2},
        {1, 2, 3},
        {-1, -2, -3},
        {4, 5, 9},
        {0, -1, -7},
        {0.5, 0.333, 1.4},
        {0, 0, 0}
    };
    arma::vec unitVec = {0,0,1};

    rotSetAroundVec3D(set, unitVec, 0.25);

    constexpr float Tol = 0.00001;
    arma::mat setTarget = {
        {-0.24740,   0.96891,   2.00000},
        {0.47410,   2.18523,   3.00000},
        {-0.47410,  -2.18523,  -3.00000},
        {2.63863,   5.83418,   9.00000},
        {0.24740,  -0.96891,  -7.00000},
        {0.40207,   0.44635,   1.40000},
        {0.00000,   0.00000,   0.00000}
    };

    // test every matrix entries - successive ROT 1
    int rowNb = set.n_rows;
    int colNb = set.n_cols;
    for (int i = 0; i < rowNb; i++)
    {
        for (int j = 0; j < colNb; j++)
        {
            EXPECT_NEAR(set(i,j), setTarget(i,j), Tol);
        }
    }

    arma::vec vec = {-1.7, 2.45, 1.2};

    rotSetAroundVec3D(set, vec, 2.64);

    setTarget = {
        {-0.82803,  1.92625, -0.77712},
        {-2.22450,  2.55790, -1.58390},
        {2.22450 , -2.55790,  1.58390},
        {-6.42613,  6.81781, -5.84999},
        {0.84839 , -5.86785,  3.85339},
        {-0.56498,  0.96693, -1.03283},
        {0.00000 ,  0.00000,  0.00000}
    };

    // test every matrix entries - successive ROT 2
    for (int i = 0; i < rowNb; i++)
    {
        for (int j = 0; j < colNb; j++)
        {
            EXPECT_NEAR(set(i,j), setTarget(i,j), Tol);
        }
    }
}

TEST(MathTests, setApplyOffset_shouldApplyOffsetToAllPointsOfASet)
{
    arma::mat set = {
        {0, 1},
        {1, 2},
        {-1, -2},
    };

    arma::vec offset = {1, -2.5};

    setApplyOffset(set, offset);

    constexpr float Tol = 0.00001;
    arma::mat setTarget = {
        {1, -1.5},
        {2, -0.5},
        {0, -4.5},
    };

    // test every matrix entries
    for (int i = 0; i < set.n_rows; i++)
    {
        for (int j = 0; j < set.n_cols; j++)
        {
            EXPECT_NEAR(set(i,j), setTarget(i,j), Tol);
        }
    }
}

TEST(MathTests, setGetCentroid_shouldGetTheCentroidOfASet)
{
    arma::mat set = {
        {0,  1},
        {1,  2},
        {-1, -2},
    };

    arma::vec centroid = setGetCentroid(set);

    constexpr float Tol = 0.00001;
    EXPECT_NEAR(centroid(0), 0, Tol);
    EXPECT_NEAR(centroid(1), 0.33333333333, Tol);
}

TEST(MathTests, findSetAngle2D_shouldReturnAngleFromXAxisAndSetOrientation)
{
    arma::mat set = {
        {0, 1},
        {1, 2},
        {-1, -2},
        {4, 5},
        {0, -1},
        {0.5, 0.333},
        {0, 0}
    };

    float angle = findSetAngle2D(set);

    constexpr float Tol = 0.00001;
    EXPECT_NEAR(angle, 0.93339, Tol);
}
