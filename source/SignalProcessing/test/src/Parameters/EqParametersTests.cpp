#include <SignalProcessing/Parameters/EqParameters.h>

#include <gtest/gtest.h>

#include <thread>

using namespace adaptone;
using namespace std;

static constexpr size_t SampleFrequency = 48000;
static constexpr double MaxAbsError = 0.0001;

TEST(EqParametersTests, biquadCoefficients_invalidChannel_shouldThrowInvalidValueException)
{
    vector<double> frequencies{ 20, 25, 50 };

    EqParameters<float> eqParameters(SampleFrequency, frequencies, 2);

    EXPECT_THROW(eqParameters.biquadCoefficients(2), InvalidValueException);
}

TEST(EqParametersTests, d0_invalidChannel_shouldThrowInvalidValueException)
{
    vector<double> frequencies{ 20, 25, 50 };

    EqParameters<float> eqParameters(SampleFrequency, frequencies, 2);

    EXPECT_THROW(eqParameters.d0(2), InvalidValueException);
}

TEST(EqParametersTests, setGraphicEqGains_invalidChannel_shouldThrowInvalidValueException)
{
    vector<double> gainsDb{ -10, 0, 10 };
    vector<double> frequencies{ 20, 25, 50 };

    EqParameters<float> eqParameters(SampleFrequency, frequencies, 2);

    EXPECT_THROW(eqParameters.setGraphicEqGains(2, gainsDb), InvalidValueException);
}

TEST(EqParametersTests, setGraphicEqGains_invalidChannel_shouldUpdateTheSpecifiedChannelDesigner)
{
    vector<double> gainsDb{ -10.1143, -10.5725, -6.013 };
    vector<double> frequencies{ 20, 50, 125 };

    EqParameters<float> eqParameters(SampleFrequency, frequencies, 2);
    EXPECT_FALSE(eqParameters.isDirty(0));
    EXPECT_FALSE(eqParameters.isDirty(1));

    eqParameters.setGraphicEqGains(0, gainsDb);
    EXPECT_TRUE(eqParameters.isDirty(0));
    EXPECT_FALSE(eqParameters.isDirty(1));

    ASSERT_EQ(eqParameters.biquadCoefficients(0).size(), 2 * frequencies.size());

    EXPECT_NEAR(eqParameters.biquadCoefficients(0)[0].b0, -2.9537e-05, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(0)[0].b1, 2.94857e-05, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(0)[0].b2, 0, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(0)[0].a1, -1.9986, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(0)[0].a2, 0.9986, MaxAbsError);

    EXPECT_NEAR(eqParameters.biquadCoefficients(0)[1].b0, -5.9236e-05, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(0)[1].b1, 5.9096e-05, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(0)[1].b2, 0, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(0)[1].a1, -1.9983, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(0)[1].a2, 0.9983, MaxAbsError);

    EXPECT_NEAR(eqParameters.biquadCoefficients(0)[2].b0, -0.0001338, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(0)[2].b1, 0.0001336, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(0)[2].b2, 0, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(0)[2].a1, -1.9980, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(0)[2].a2, 0.9980, MaxAbsError);

    EXPECT_NEAR(eqParameters.d0(0), 0.3876, MaxAbsError);

    ASSERT_EQ(eqParameters.biquadCoefficients(1).size(), 2 * frequencies.size());

    EXPECT_NEAR(eqParameters.biquadCoefficients(1)[0].b0, 0, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(1)[0].b1, 0, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(1)[0].b2, 0, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(1)[0].a1, -1.9986, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(1)[0].a2, 0.9986, MaxAbsError);

    EXPECT_NEAR(eqParameters.biquadCoefficients(1)[1].b0, 0, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(1)[1].b1, 0, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(1)[1].b2, 0, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(1)[1].a1, -1.9983, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(1)[1].a2, 0.9983, MaxAbsError);

    EXPECT_NEAR(eqParameters.biquadCoefficients(1)[2].b0, 0, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(1)[2].b1, 0, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(1)[2].b2, 0, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(1)[2].a1, -1.9980, MaxAbsError);
    EXPECT_NEAR(eqParameters.biquadCoefficients(1)[2].a2, 0.9980, MaxAbsError);

    EXPECT_NEAR(eqParameters.d0(1), 1, MaxAbsError);
}

TEST(EqParametersTests, applyUpdate_shouldExecuteTheFunctionOnlyIfTheParametersAreDirtyAndSetDirtyToFalse)
{
    vector<double> gainsDb{ -10.1143, -10.5725, -6.013 };
    vector<double> frequencies{ 20, 50, 125 };

    EqParameters<float> eqParameters(SampleFrequency, frequencies, 2);

    int counter = 0;
    auto function = [&]()
    {
        counter++;
    };

    eqParameters.applyUpdate(0, function);
    EXPECT_EQ(counter, 0);
    EXPECT_FALSE(eqParameters.isDirty(0));

    eqParameters.applyUpdate(1, function);
    EXPECT_EQ(counter, 0);
    EXPECT_FALSE(eqParameters.isDirty(1));

    eqParameters.setGraphicEqGains(0, gainsDb);
    EXPECT_TRUE(eqParameters.isDirty(0));
    EXPECT_FALSE(eqParameters.isDirty(1));

    eqParameters.applyUpdate(0, function);
    eqParameters.applyUpdate(0, function);
    EXPECT_EQ(counter, 1);
    EXPECT_FALSE(eqParameters.isDirty(0));
    EXPECT_FALSE(eqParameters.isDirty(1));

    eqParameters.setGraphicEqGains(1, gainsDb);
    EXPECT_FALSE(eqParameters.isDirty(0));
    EXPECT_TRUE(eqParameters.isDirty(1));

    eqParameters.applyUpdate(1, function);
    eqParameters.applyUpdate(1, function);
    EXPECT_EQ(counter, 2);
    EXPECT_FALSE(eqParameters.isDirty(0));
    EXPECT_FALSE(eqParameters.isDirty(1));
}
