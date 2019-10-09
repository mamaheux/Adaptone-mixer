#include <Uniformization/AutoPosition.h>
#include <Uniformization/Room.h>

#include <gtest/gtest.h>
#include <armadillo>

using namespace adaptone;
using namespace std;

TEST(AutoPositionTests, computeRoomConfiguration2D_shouldFindProperRoomConfiguration)
{
    constexpr float Tol = 0.5;

    constexpr int TestNb = 5;

    constexpr double Alpha = 0.75;
    constexpr double EpsilonTotalDistError = 5e-4;
    constexpr double EpsilonDeltaTotalDistError = 1e-6;

    constexpr int IterNb = 5000;
    constexpr int ThermalIterNb = 200;
    constexpr int TryNb = 10;
    constexpr int CountThreshold = 10;

    constexpr float DistRelativeError = 0;

    arma::mat speakersMat = {
        {0,0},
        {1,-0.5},
        {2,0},
        {3,-0.5},
        {4,0}
    };
    arma::mat probesMat = {
        {0,3},
        {1,4},
        {1.5,2},
        {2.5,2},
        {3,4},
        {4,3}
    };
    int speakerNb = speakersMat.n_rows;
    int probeNb = probesMat.n_rows;

    for(int n = 0; n < TestNb; n++)
    {
        arma::mat distMat = arma::zeros<arma::mat>(speakerNb, probeNb);

        // compute distance between each pair of the two set (A and B)
        for (int i = 0; i < speakerNb; i++)
        {
            for (int j = 0; j < probeNb; j++)
            {
                distMat(i, j) = arma::norm(speakersMat.row(i) - probesMat.row(j));
            }
        }

        arma::mat speakersNewMat = arma::randu<arma::mat>(speakerNb, 3);
        arma::mat probesNewMat = arma::randu<arma::mat>(probeNb, 3);

        Room room = Room(speakersNewMat, probesNewMat);

        AutoPosition autoPos = AutoPosition(Alpha, EpsilonTotalDistError, EpsilonDeltaTotalDistError, IterNb,
            ThermalIterNb, TryNb, CountThreshold);

        autoPos.computeRoomConfiguration2D(room, distMat, DistRelativeError, true);

        speakersNewMat = room.getSpeakersPosMat();
        probesNewMat = room.getProbesPosMat();

        arma::mat speakersTargetMat = {
            {-2, 0.2, 0},
            {-1, -0.3, 0},
            {0, 0.2, 0},
            {1, -0.3, 0},
            {2, 0.2, 0}
        };
        arma::mat probesTargetMat = {
            {-2, 3.2, 0},
            {-1, 4.2, 0},
            {-0.5, 2.2, 0},
            {0.5, 2.2, 0},
            {1, 4.2, 0},
            {2, 3.2, 0}
        };

        speakersNewMat.save("test1A.txt", arma::csv_ascii);
        probesNewMat.save("test2A.txt", arma::csv_ascii);

        // check if the configuration is a symmetry of the target configuration
        if (abs(speakersNewMat(0,0) - speakersTargetMat(0,0)) > Tol)
        {
            speakersNewMat.col(0) *= -1;
            probesNewMat.col(0) *= -1;
        }

        speakersNewMat.save("test1.txt", arma::csv_ascii);
        probesNewMat.save("test2.txt", arma::csv_ascii);

        // Check correct position for each Probes and Speakers
        for (int i = 0; i < speakerNb; i++)
        {
            EXPECT_NEAR(speakersNewMat(i,0), speakersTargetMat(i,0), Tol);
            EXPECT_NEAR(speakersNewMat(i,1), speakersTargetMat(i,1), Tol);
            EXPECT_NEAR(speakersNewMat(i,2), speakersTargetMat(i,2), Tol);
        }
        for (int i = 0; i < probeNb; i++)
        {
            EXPECT_NEAR(probesNewMat(i,0), probesTargetMat(i,0), Tol);
            EXPECT_NEAR(probesNewMat(i,1), probesTargetMat(i,1), Tol);
            EXPECT_NEAR(probesNewMat(i,2), probesTargetMat(i,2), Tol);
        }
    }
}

