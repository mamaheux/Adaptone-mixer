#include <Uniformization/Model/Probe.h>

#include <gtest/gtest.h>

using namespace adaptone;

TEST(ProbeTests, constructor_default_shouldSetCoordinatesTo0)
{
    Probe probe;

    EXPECT_EQ(probe.x(), 0);
    EXPECT_EQ(probe.y(), 0);
    EXPECT_EQ(probe.z(), 0);
}

TEST(ProbeTests, constructor_xy_shouldSetTheCoordinates)
{
    constexpr double X = 10;
    constexpr double Y = 15;
    Probe probe(X, Y);

    EXPECT_EQ(probe.x(), X);
    EXPECT_EQ(probe.y(), Y);
    EXPECT_EQ(probe.z(), 0);
}

TEST(ProbeTests, constructor_xyz_shouldSetTheCoordinates)
{
    constexpr double X = 10;
    constexpr double Y = 15;
    constexpr double Z = 20;
    Probe probe(X, Y, Z);

    EXPECT_EQ(probe.x(), X);
    EXPECT_EQ(probe.y(), Y);
    EXPECT_EQ(probe.z(), Z);
}
