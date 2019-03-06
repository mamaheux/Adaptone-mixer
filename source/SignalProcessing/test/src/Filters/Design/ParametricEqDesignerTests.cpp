#include <SignalProcessing/Filters/Design/ParametricEqDesigner.h>

#include <gtest/gtest.h>

#include <iostream>
#include <chrono>

using namespace adaptone;
using namespace std;

TEST(ParametricEqDesignerTests, constructor_invalidFilterCount_shouldThrowInvalidValueException)
{
    EXPECT_THROW(ParametricEqDesigner<float>(1, 48000), InvalidValueException);
}

TEST(ParametricEqDesignerTests, constructor_invalidParameterCount_shouldThrowInvalidValueException)
{
    ParametricEqDesigner<float> designer(2, 48000);

    vector<ParametricEqDesigner<float>::Parameters> parameters
        {
            ParametricEqDesigner<float>::Parameters(100, 1, -10),
            ParametricEqDesigner<float>::Parameters(300, 5, 5),
            ParametricEqDesigner<float>::Parameters(800, 5, -8)
        };

    EXPECT_THROW(designer.update(parameters), InvalidValueException);
}

TEST(ParametricEqDesignerTests, update_lowShelveGainLessThan0_highShelveGainGreaterThan0_shouldSetTheRightCoefficents)
{
    ParametricEqDesigner<double> designer(5, 48000);

    vector<ParametricEqDesigner<double>::Parameters> parameters
        {
            ParametricEqDesigner<double>::Parameters(100, 1, -10),
            ParametricEqDesigner<double>::Parameters(300, 5, 5),
            ParametricEqDesigner<double>::Parameters(800, 5, -8),
            ParametricEqDesigner<double>::Parameters(1500, 5, 12),
            ParametricEqDesigner<double>::Parameters(8000, 1, 2),
        };

    designer.update(parameters);

    EXPECT_EQ(designer.biquadCoefficients().size(), 5);

    EXPECT_NEAR(designer.biquadCoefficients()[0].b0, 0.9949, 0.0001);
    EXPECT_NEAR(designer.biquadCoefficients()[0].b1, -1.9766, 0.0001);
    EXPECT_NEAR(designer.biquadCoefficients()[0].b2, 0.9819, 0.0001);
    EXPECT_NEAR(designer.biquadCoefficients()[0].a1, -1.9765, 0.0001);
    EXPECT_NEAR(designer.biquadCoefficients()[0].a2, 0.9770, 0.0001);

    EXPECT_NEAR(designer.biquadCoefficients()[1].b0, 1.0044, 0.0001);
    EXPECT_NEAR(designer.biquadCoefficients()[1].b1, -1.9872, 0.0001);
    EXPECT_NEAR(designer.biquadCoefficients()[1].b2, 0.9844, 0.0001);
    EXPECT_NEAR(designer.biquadCoefficients()[1].a1, -1.9872, 0.0001);
    EXPECT_NEAR(designer.biquadCoefficients()[1].a2, 0.9888, 0.0001);

    EXPECT_NEAR(designer.biquadCoefficients()[2].b0, 0.9825, 0.0001);
    EXPECT_NEAR(designer.biquadCoefficients()[2].b1, -1.9312, 0.0001);
    EXPECT_NEAR(designer.biquadCoefficients()[2].b2, 0.9593, 0.0001);
    EXPECT_NEAR(designer.biquadCoefficients()[2].a1, -1.9312, 0.0001);
    EXPECT_NEAR(designer.biquadCoefficients()[2].a2, 0.9418, 0.0001);

    EXPECT_NEAR(designer.biquadCoefficients()[3].b0, 1.0463, 0.0001);
    EXPECT_NEAR(designer.biquadCoefficients()[3].b1, -1.9311, 0.0001);
    EXPECT_NEAR(designer.biquadCoefficients()[3].b2, 0.9227, 0.0001);
    EXPECT_NEAR(designer.biquadCoefficients()[3].a1, -1.9311, 0.0001);
    EXPECT_NEAR(designer.biquadCoefficients()[3].a2, 0.9690, 0.0001);

    EXPECT_NEAR(designer.biquadCoefficients()[4].b0, 1.0820, 0.0001);
    EXPECT_NEAR(designer.biquadCoefficients()[4].b1, -0.6075, 0.0001);
    EXPECT_NEAR(designer.biquadCoefficients()[4].b2, 0.4040, 0.0001);
    EXPECT_NEAR(designer.biquadCoefficients()[4].a1, -0.6978, 0.0001);
    EXPECT_NEAR(designer.biquadCoefficients()[4].a2, 0.3957, 0.0001);
}

TEST(ParametricEqDesignerTests, gainsDb_shouldReturnTheGainAtTheSpecifiedFrequencies)
{
    ParametricEqDesigner<double> designer(5, 48000);

    vector<ParametricEqDesigner<double>::Parameters> parameters
        {
            ParametricEqDesigner<double>::Parameters(100, 1, -10),
            ParametricEqDesigner<double>::Parameters(300, 5, 5),
            ParametricEqDesigner<double>::Parameters(800, 5, -8),
            ParametricEqDesigner<double>::Parameters(1500, 5, 12),
            ParametricEqDesigner<double>::Parameters(8000, 1, 2),
        };

    designer.update(parameters);

    vector<double> frequencies{ 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
        1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000 };

    vector<double> gainsDb = designer.gainsDb(frequencies);

    EXPECT_EQ(gainsDb.size(), frequencies.size());

    EXPECT_NEAR(gainsDb[0], -8.1124, 0.0001);
    EXPECT_NEAR(gainsDb[1], -8.1725, 0.0001);
    EXPECT_NEAR(gainsDb[2], -8.2649, 0.0001);
    EXPECT_NEAR(gainsDb[3], -8.4005, 0.0001);
    EXPECT_NEAR(gainsDb[4], -8.5535, 0.0001);
    EXPECT_NEAR(gainsDb[5], -8.6506, 0.0001);
    EXPECT_NEAR(gainsDb[6], -8.3168, 0.0001);
    EXPECT_NEAR(gainsDb[7], -6.8466, 0.0001);
    EXPECT_NEAR(gainsDb[8], -3.8406, 0.0001);
    EXPECT_NEAR(gainsDb[9], 0.0749, 0.0001);
    EXPECT_NEAR(gainsDb[10], 2.7331, 0.0001);
    EXPECT_NEAR(gainsDb[11], 5.1438, 0.0001);
    EXPECT_NEAR(gainsDb[12], 7.1641, 0.0001);
    EXPECT_NEAR(gainsDb[13], 3.6670, 0.0001);
    EXPECT_NEAR(gainsDb[14], 2.1088, 0.0001);
    EXPECT_NEAR(gainsDb[15], -0.0263, 0.0001);
    EXPECT_NEAR(gainsDb[16], -4.8773, 0.0001);
    EXPECT_NEAR(gainsDb[17], 0.8581, 0.0001);
    EXPECT_NEAR(gainsDb[18], 6.2804, 0.0001);
    EXPECT_NEAR(gainsDb[19], 11.6116, 0.0001);
    EXPECT_NEAR(gainsDb[20], 4.9273, 0.0001);
    EXPECT_NEAR(gainsDb[21], 3.1708, 0.0001);
    EXPECT_NEAR(gainsDb[22], 2.6034, 0.0001);
    EXPECT_NEAR(gainsDb[23], 2.3932, 0.0001);
    EXPECT_NEAR(gainsDb[24], 2.2907, 0.0001);
    EXPECT_NEAR(gainsDb[25], 2.0593, 0.0001);
    EXPECT_NEAR(gainsDb[26], 1.2691, 0.0001);
    EXPECT_NEAR(gainsDb[27], 0.2533, 0.0001);
    EXPECT_NEAR(gainsDb[28], -0.1308, 0.0001);
    EXPECT_NEAR(gainsDb[29], -0.1005, 0.0001);
    EXPECT_NEAR(gainsDb[30], -0.0252, 0.0001);
}

TEST(ParametricEqDesignerTests, performance)
{
    ParametricEqDesigner<double> designer(5, 48000);

    vector<ParametricEqDesigner<double>::Parameters> parameters
        {
            ParametricEqDesigner<double>::Parameters(100, 1, -10),
            ParametricEqDesigner<double>::Parameters(300, 5, 5),
            ParametricEqDesigner<double>::Parameters(800, 5, -8),
            ParametricEqDesigner<double>::Parameters(1500, 5, 12),
            ParametricEqDesigner<double>::Parameters(8000, 1, 2),
        };

    vector<double> frequencies{ 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
        1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000 };

    constexpr std::size_t Count = 10000;

    auto start = chrono::system_clock::now();
    for (std::size_t i = 0; i < Count; i++)
    {
        designer.update(parameters);
        vector<double> gainsDb = designer.gainsDb(frequencies);
    }
    auto end = chrono::system_clock::now();
    chrono::duration<double> elapsed_seconds = end - start;

    cout << "Elapsed time = " << elapsed_seconds.count() / Count << " s" << endl;
}
