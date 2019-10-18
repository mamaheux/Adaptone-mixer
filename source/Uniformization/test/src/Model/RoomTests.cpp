#include <Uniformization/Model/Room.h>
#include <UniformizationTests/ArmadilloUtils.h>
#include <Utils/Exception/InvalidValueException.h>

#include <armadillo>

#include <gtest/gtest.h>

using namespace adaptone;

TEST(RoomTests, constructor_shouldSetTheDefaultCoordinates)
{
    constexpr double Tolerance = 0.00001;
    constexpr double SpeakerCount = 2;
    constexpr double ProbeCount = 3;
    Room room(SpeakerCount, ProbeCount);

    arma::mat speakersPosMat = room.getSpeakersPosMat();
    arma::mat probesPosMat = room.getProbesPosMat();

    const arma::mat SpeakersPosMatTarget =
    {
        { 0, 0, 0 },
        { 1, 0, 0 },
    };
    const arma::mat ProbesPosMatTarget =
    {
        { 0, 5, 0 },
        { 1, 5, 0 },
        { 2, 5, 0 },
    };

    EXPECT_MAT_NEAR(speakersPosMat, SpeakersPosMatTarget, Tolerance);
    EXPECT_MAT_NEAR(probesPosMat, ProbesPosMatTarget, Tolerance);
}

TEST(RoomTests, constructor_coordMat_shouldSetTheProperCoordinates)
{
    constexpr double Tolerance = 0.00001;
    const arma::mat SpeakersPosMatTarget =
        {
            { 0, 0, 1 },
            { 1, 0, 2 },
        };
    const arma::mat ProbesPosMatTarget =
        {
            { 0, 5, 1 },
            { 1, 5, 2 },
            { 2, 5, 3 },
        };

    Room room(SpeakersPosMatTarget, ProbesPosMatTarget);

    arma::mat speakersPosMat = room.getSpeakersPosMat();
    arma::mat probesPosMat = room.getProbesPosMat();

    EXPECT_MAT_NEAR(speakersPosMat, SpeakersPosMatTarget, Tolerance);
    EXPECT_MAT_NEAR(probesPosMat, ProbesPosMatTarget, Tolerance);
}

TEST(RoomTests, setProbesPosFromMat_setSpeakersPosFromMat_shouldSetTheProperProbesCoordinates)
{
    constexpr double Tolerance = 0.00001;
    constexpr double SpeakerCount = 2;
    constexpr double ProbeCount = 3;
    const arma::mat SpeakersPosMatTarget =
        {
            { 0, 0, 1 },
            { 1, 0, 2 },
        };
    const arma::mat ProbesPosMatTarget =
        {
            { 0, 5, 1 },
            { 1, 5, 2 },
            { 2, 5, 3 },
        };

    Room room(SpeakerCount, ProbeCount);

    room.setSpeakersPosFromMat(SpeakersPosMatTarget);
    room.setProbesPosFromMat(ProbesPosMatTarget);

    arma::mat speakersPosMat = room.getSpeakersPosMat();
    arma::mat probesPosMat = room.getProbesPosMat();

    EXPECT_MAT_NEAR(speakersPosMat, SpeakersPosMatTarget, Tolerance);
    EXPECT_MAT_NEAR(probesPosMat, ProbesPosMatTarget, Tolerance);
}

TEST(RoomTests, setProbesPosFromMat_setSpeakersPosFromMat_checkForInvalidInputMatrixFormat)
{
    constexpr double Tolerance = 0.00001;
    constexpr double SpeakerCount = 2;
    constexpr double ProbeCount = 3;
    const arma::mat SpeakersPosMatTarget =
        {
            { 0, 0, 1 },
            { 1, 0, 2 },
            { 0, 0, 0,}
        };
    const arma::mat ProbesPosMatTarget =
        {
            { 0, 5, 1 },
            { 1, 5, 2 }
        };

    Room room(SpeakerCount, ProbeCount);

    EXPECT_THROW(room.setSpeakersPosFromMat(SpeakersPosMatTarget), InvalidValueException);
    EXPECT_THROW(room.setProbesPosFromMat(ProbesPosMatTarget), InvalidValueException);
}
