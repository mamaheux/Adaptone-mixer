#include <Uniformization/Model/AutoPosition.h>
#include <Uniformization/Model/Room.h>

#include <gtest/gtest.h>
#include <armadillo>

using namespace adaptone;
using namespace std;

TEST(AutoPositionTests, computeRoomConfiguration2D_shouldFindProperRoomConfiguration)
{
    constexpr float Tolerance = 0.5;

    constexpr int TestCount = 5;

    constexpr double Alpha = 1.0;
    constexpr double EpsilonTotalDistanceError = 5e-4;
    constexpr double EpsilonDeltaTotalDistanceError = 1e-6;

    constexpr int IterationCount = 5000;
    constexpr int ThermalIterationCount = 500;
    constexpr int TryCount = 10;
    constexpr int CountThreshold = 10;

    constexpr float DistanceRelativeError = 0;

    arma::mat speakersMat = {
        { 0, 0 },
        { 1, -0.5 },
        { 2, 0 },
        { 3, -0.5 },
        { 4, 0 }
    };
    arma::mat probesMat = {
        { 0, 3 },
        { 1, 4 },
        { 1.5, 2 },
        { 2.5, 2 },
        { 3, 4 },
        { 4, 3 }
    };
    int speakerCount = speakersMat.n_rows;
    int probeCount = probesMat.n_rows;

    for(int n = 0; n < TestCount; n++)
    {
        arma::mat distancesMat = arma::zeros<arma::mat>(speakerCount, probeCount);

        // compute distance between each pair of the two set (A and B)
        for (int i = 0; i < speakerCount; i++)
        {
            for (int j = 0; j < probeCount; j++)
            {
                distancesMat(i, j) = arma::norm(speakersMat.row(i) - probesMat.row(j));
            }
        }

        arma::mat speakersNewMat = arma::randu<arma::mat>(speakerCount, 3);
        arma::mat probesNewMat = arma::randu<arma::mat>(probeCount, 3);

        Room room = Room(speakersNewMat, probesNewMat);

        AutoPosition autoPos = AutoPosition(Alpha, EpsilonTotalDistanceError, EpsilonDeltaTotalDistanceError, IterationCount,
            ThermalIterationCount, TryCount, CountThreshold);

        autoPos.computeRoomConfiguration2D(room, distancesMat, DistanceRelativeError, true);

        speakersNewMat = room.getSpeakersPosMat();
        probesNewMat = room.getProbesPosMat();

        arma::mat speakersTargetMat = {
            { -2, 0.2, 0 },
            { -1, -0.3, 0 },
            { 0, 0.2, 0 },
            { 1, -0.3, 0 },
            { 2, 0.2, 0 }
        };
        arma::mat probesTargetMat = {
            { -2, 3.2, 0 },
            { -1, 4.2, 0 },
            { -0.5, 2.2, 0 },
            { 0.5, 2.2, 0 },
            { 1, 4.2, 0 },
            { 2, 3.2, 0 }
        };

        speakersNewMat.save("test1A.txt", arma::csv_ascii);
        probesNewMat.save("test2A.txt", arma::csv_ascii);

        // check if the configuration is a symmetry of the target configuration
        if (abs(speakersNewMat(0,0) - speakersTargetMat(0,0)) > Tolerance)
        {
            speakersNewMat.col(0) *= -1;
            probesNewMat.col(0) *= -1;
        }

        speakersNewMat.save("test1.txt", arma::csv_ascii);
        probesNewMat.save("test2.txt", arma::csv_ascii);

        // Check correct position for each Probes and Speakers
        for (int i = 0; i < speakerCount; i++)
        {
            EXPECT_NEAR(speakersNewMat(i,0), speakersTargetMat(i,0), Tolerance);
            EXPECT_NEAR(speakersNewMat(i,1), speakersTargetMat(i,1), Tolerance);
            EXPECT_NEAR(speakersNewMat(i,2), speakersTargetMat(i,2), Tolerance);
        }
        for (int i = 0; i < probeCount; i++)
        {
            EXPECT_NEAR(probesNewMat(i,0), probesTargetMat(i,0), Tolerance);
            EXPECT_NEAR(probesNewMat(i,1), probesTargetMat(i,1), Tolerance);
            EXPECT_NEAR(probesNewMat(i,2), probesTargetMat(i,2), Tolerance);
        }
    }
}

