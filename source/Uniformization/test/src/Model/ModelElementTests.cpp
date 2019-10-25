#include <Uniformization/Model/ModelElement.h>

#include <gtest/gtest.h>

using namespace adaptone;

TEST(ModelElementTests, constructor_default_shouldSetCoordinatesTo0)
{
    ModelElement modelElement;

    EXPECT_EQ(modelElement.x(), 0);
    EXPECT_EQ(modelElement.y(), 0);
    EXPECT_EQ(modelElement.z(), 0);
    EXPECT_EQ(modelElement.id(), 0);
}

TEST(ModelElementTests, constructor_xy_shouldSetTheCoordinates)
{
    constexpr double X = 10;
    constexpr double Y = 15;
    constexpr uint32_t Id = 5;
    ModelElement modelElement(X, Y, Id);

    EXPECT_EQ(modelElement.x(), X);
    EXPECT_EQ(modelElement.y(), Y);
    EXPECT_EQ(modelElement.z(), 0);
    EXPECT_EQ(modelElement.id(), Id);
}

TEST(ModelElementTests, constructor_xyz_shouldSetTheCoordinates)
{
    constexpr double X = 10;
    constexpr double Y = 15;
    constexpr double Z = 20;
    constexpr uint32_t Id = 5;
    ModelElement modelElement(X, Y, Z, Id);

    EXPECT_EQ(modelElement.x(), X);
    EXPECT_EQ(modelElement.y(), Y);
    EXPECT_EQ(modelElement.z(), Z);
    EXPECT_EQ(modelElement.id(), Id);
}

TEST(ModelElementTests, set_shouldSetTheCoordinates)
{
    constexpr double X = 10;
    constexpr double Y = 15;
    constexpr double Z = 20;
    constexpr uint32_t Id = 5;
    ModelElement modelElement;
    modelElement.setX(X);
    modelElement.setY(Y);
    modelElement.setZ(Z);
    modelElement.setId(Id);

    EXPECT_EQ(modelElement.x(), X);
    EXPECT_EQ(modelElement.y(), Y);
    EXPECT_EQ(modelElement.z(), Z);
    EXPECT_EQ(modelElement.id(), Id);
}
