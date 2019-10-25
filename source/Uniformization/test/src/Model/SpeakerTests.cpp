#include <Uniformization/Model/Speaker.h>

#include <gtest/gtest.h>

using namespace adaptone;

TEST(SpeakerTests, constructor_default_shouldSetCoordinatesTo0)
{
    Speaker speaker;

    EXPECT_EQ(speaker.x(), 0);
    EXPECT_EQ(speaker.y(), 0);
    EXPECT_EQ(speaker.z(), 0);
    EXPECT_EQ(speaker.id(), 0);
}

TEST(SpeakerTests, constructor_xy_shouldSetTheCoordinates)
{
    constexpr double X = 10;
    constexpr double Y = 15;
    constexpr uint32_t Id = 5;
    Speaker speaker(X, Y, Id);

    EXPECT_EQ(speaker.x(), X);
    EXPECT_EQ(speaker.y(), Y);
    EXPECT_EQ(speaker.z(), 0);
    EXPECT_EQ(speaker.id(), Id);
}

TEST(SpeakerTests, constructor_xyz_shouldSetTheCoordinates)
{
    constexpr double X = 10;
    constexpr double Y = 15;
    constexpr double Z = 20;
    constexpr uint32_t Id = 5;
    Speaker speaker(X, Y, Z, Id);

    EXPECT_EQ(speaker.x(), X);
    EXPECT_EQ(speaker.y(), Y);
    EXPECT_EQ(speaker.z(), Z);
    EXPECT_EQ(speaker.id(), Id);
}
