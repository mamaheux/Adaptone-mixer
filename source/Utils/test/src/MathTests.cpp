#include <Utils/Math.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(MathTests, scalarToDb_shouldConvertTheSpecifiedValueToDb)
{
    EXPECT_EQ(scalarToDb(1.0), 0.0);
    EXPECT_EQ(scalarToDb(10.0), 20.0);
    EXPECT_EQ(scalarToDb(100.0), 40.0);
    EXPECT_EQ(scalarToDb(1000.0), 60.0);
}

TEST(MathTests, vectorToDb_shouldConvertTheSpecifiedValueToDb)
{
    EXPECT_EQ(vectorToDb(vector<double>({ 1, 10, 100 })), vector<double>({ 0, 20, 40 }));
}
