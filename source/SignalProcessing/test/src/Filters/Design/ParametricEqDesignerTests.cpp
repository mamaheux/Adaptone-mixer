#include <SignalProcessing/Filters/Design/ParametricEqDesigner.h>

#include <gtest/gtest.h>

#include <iostream>
#include <chrono>

using namespace adaptone;
using namespace std;

static constexpr size_t SampleFrequency = 48000;
static constexpr double MaxAbsError = 0.0001;

TEST(ParametricEqDesignerTests, constructor_invalidFilterCount_shouldThrowInvalidValueException)
{
    EXPECT_THROW(ParametricEqDesigner<float>(1, SampleFrequency), InvalidValueException);
}

TEST(ParametricEqDesignerTests, update_invalidParameterCount_shouldThrowInvalidValueException)
{
    ParametricEqDesigner<float> designer(2, SampleFrequency);

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
    ParametricEqDesigner<double> designer(5, SampleFrequency);

    vector<ParametricEqDesigner<double>::Parameters> parameters
        {
            ParametricEqDesigner<double>::Parameters(100, 1, -10),
            ParametricEqDesigner<double>::Parameters(300, 5, 5),
            ParametricEqDesigner<double>::Parameters(800, 5, -8),
            ParametricEqDesigner<double>::Parameters(1500, 5, 12),
            ParametricEqDesigner<double>::Parameters(8000, 1, 2),
        };

    designer.update(parameters);

    ASSERT_EQ(designer.biquadCoefficients().size(), 5);

    EXPECT_NEAR(designer.biquadCoefficients()[0].b0, 0.9949, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[0].b1, -1.9766, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[0].b2, 0.9819, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[0].a1, -1.9765, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[0].a2, 0.9770, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[1].b0, 1.0044, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[1].b1, -1.9872, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[1].b2, 0.9844, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[1].a1, -1.9872, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[1].a2, 0.9888, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[2].b0, 0.9825, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[2].b1, -1.9312, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[2].b2, 0.9593, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[2].a1, -1.9312, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[2].a2, 0.9418, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[3].b0, 1.0463, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[3].b1, -1.9311, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[3].b2, 0.9227, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[3].a1, -1.9311, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[3].a2, 0.9690, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[4].b0, 1.1724, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[4].b1, -0.9689, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[4].b2, 0.4943, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[4].a1, -0.6978, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[4].a2, 0.3957, MaxAbsError);
}

TEST(ParametricEqDesignerTests, update_lowShelveGainGreaterThan0_highShelveGainLessThan0_shouldSetTheRightCoefficents)
{
    ParametricEqDesigner<float> designer(2, SampleFrequency);

    vector<ParametricEqDesigner<float>::Parameters> parameters
        {
            ParametricEqDesigner<float>::Parameters(50, 1, 12),
            ParametricEqDesigner<float>::Parameters(12000, 1, -5),
        };

    designer.update(parameters);

    ASSERT_EQ(designer.biquadCoefficients().size(), 2);

    EXPECT_NEAR(designer.biquadCoefficients()[0].b0, 1.0033, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[0].b1, -1.9934, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[0].b2, 0.9903, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[0].a1, -1.9934, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[0].a2, 0.9935, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[1].b0, 0.7296, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[1].b1, 0, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[1].b2, 0.2432, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[1].a1, -0.3786, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[1].a2, 0.3514, MaxAbsError);
}

TEST(ParametricEqDesignerTests, update_lowShelveGain0_highShelveGain0_shouldSetTheRightCoefficents)
{
    ParametricEqDesigner<double> designer(2, SampleFrequency);

    vector<ParametricEqDesigner<double>::Parameters> parameters
        {
            ParametricEqDesigner<double>::Parameters(100, 1, 0),
            ParametricEqDesigner<double>::Parameters(12000, 1, 0),
        };

    designer.update(parameters);

    ASSERT_EQ(designer.biquadCoefficients().size(), 2);

    EXPECT_NEAR(designer.biquadCoefficients()[0].b0, 1, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[0].b1, 0, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[0].b2, 0, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[0].a1, 0, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[0].a2, 0, MaxAbsError);

    EXPECT_NEAR(designer.biquadCoefficients()[1].b0, 1, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[1].b1, 0, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[1].b2, 0, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[1].a1, 0, MaxAbsError);
    EXPECT_NEAR(designer.biquadCoefficients()[1].a2, 0, MaxAbsError);
}


TEST(ParametricEqDesignerTests, gainsDb_shouldReturnTheGainAtTheSpecifiedFrequencies)
{
    ParametricEqDesigner<double> designer(5, SampleFrequency);

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

    EXPECT_NEAR(gainsDb[0], -10.1124, MaxAbsError);
    EXPECT_NEAR(gainsDb[1], -10.1725, MaxAbsError);
    EXPECT_NEAR(gainsDb[2], -10.2650, MaxAbsError);
    EXPECT_NEAR(gainsDb[3], -10.4006, MaxAbsError);
    EXPECT_NEAR(gainsDb[4], -10.5536, MaxAbsError);
    EXPECT_NEAR(gainsDb[5], -10.6507, MaxAbsError);
    EXPECT_NEAR(gainsDb[6], -10.3170, MaxAbsError);
    EXPECT_NEAR(gainsDb[7], -8.8469, MaxAbsError);
    EXPECT_NEAR(gainsDb[8], -5.8410, MaxAbsError);
    EXPECT_NEAR(gainsDb[9], -1.9258, MaxAbsError);
    EXPECT_NEAR(gainsDb[10], 0.7320, MaxAbsError);
    EXPECT_NEAR(gainsDb[11], 3.1422, MaxAbsError);
    EXPECT_NEAR(gainsDb[12], 5.1615, MaxAbsError);
    EXPECT_NEAR(gainsDb[13], 1.6629, MaxAbsError);
    EXPECT_NEAR(gainsDb[14], 0.1023, MaxAbsError);
    EXPECT_NEAR(gainsDb[15], -2.0365, MaxAbsError);
    EXPECT_NEAR(gainsDb[16], -6.8938, MaxAbsError);
    EXPECT_NEAR(gainsDb[17], -1.1675, MaxAbsError);
    EXPECT_NEAR(gainsDb[18], 4.2406, MaxAbsError);
    EXPECT_NEAR(gainsDb[19], 9.5472, MaxAbsError);
    EXPECT_NEAR(gainsDb[20], 2.8286, MaxAbsError);
    EXPECT_NEAR(gainsDb[21], 1.0221, MaxAbsError);
    EXPECT_NEAR(gainsDb[22], 0.3847, MaxAbsError);
    EXPECT_NEAR(gainsDb[23], 0.1011, MaxAbsError);
    EXPECT_NEAR(gainsDb[24], 0.0303, MaxAbsError);
    EXPECT_NEAR(gainsDb[25], 0.3259, MaxAbsError);
    EXPECT_NEAR(gainsDb[26], 1.2691, MaxAbsError);
    EXPECT_NEAR(gainsDb[27], 2.0135, MaxAbsError);
    EXPECT_NEAR(gainsDb[28], 2.1661, MaxAbsError);
    EXPECT_NEAR(gainsDb[29], 2.0919, MaxAbsError);
    EXPECT_NEAR(gainsDb[30], 2.0219, MaxAbsError);
}

TEST(ParametricEqDesignerTests, performance)
{
    ParametricEqDesigner<double> designer(5, SampleFrequency);

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
    chrono::duration<double> elapsedSeconds = end - start;

    cout << "Elapsed time = " << elapsedSeconds.count() / Count << " s" << endl;
}
