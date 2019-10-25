#include <Uniformization/Model/Probe.h>

#include <gtest/gtest.h>

using namespace adaptone;

TEST(ProbeTests, constructor_default_shouldSetCoordinatesTo0)
{
    Probe probe;

    EXPECT_EQ(probe.x(), 0);
    EXPECT_EQ(probe.y(), 0);
    EXPECT_EQ(probe.z(), 0);
    EXPECT_EQ(probe.id(), 0);
}

TEST(ProbeTests, constructor_xy_shouldSetTheCoordinates)
{
    constexpr double X = 10;
    constexpr double Y = 15;
    constexpr uint32_t Id = 5;
    Probe probe(X, Y, Id);

    EXPECT_EQ(probe.x(), X);
    EXPECT_EQ(probe.y(), Y);
    EXPECT_EQ(probe.z(), 0);
    EXPECT_EQ(probe.id(), Id);
}

TEST(ProbeTests, constructor_xyz_shouldSetTheCoordinates)
{
    constexpr double X = 10;
    constexpr double Y = 15;
    constexpr double Z = 20;
    constexpr uint32_t Id = 5;
    Probe probe(X, Y, Z, Id);

    EXPECT_EQ(probe.x(), X);
    EXPECT_EQ(probe.y(), Y);
    EXPECT_EQ(probe.z(), Z);
    EXPECT_EQ(probe.id(), Id);
}
