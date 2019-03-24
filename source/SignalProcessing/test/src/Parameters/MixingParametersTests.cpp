#include <SignalProcessing/Parameters/MixingParameters.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(MixingParametersTests, constructor_shouldInitializeAllGainsTo0)
{
    MixingParameters<float> mixingParameters(2, 2);

    EXPECT_EQ(mixingParameters.gains(), vector<float>({ 0, 0, 0, 0 }));
}

TEST(MixingParametersTests, setGain_invalidChannel_shouldThrowInvalidException)
{
    MixingParameters<float> mixingParameters(2, 3);

    EXPECT_THROW(mixingParameters.setGain(2, 1, 20), InvalidValueException);
    EXPECT_THROW(mixingParameters.setGain(1, 3, 20), InvalidValueException);
    EXPECT_THROW(mixingParameters.setGain(2, 3, 20), InvalidValueException);
}

TEST(MixingParametersTests, setGain_shouldSetSpecifiedChannelGainNotInDb)
{
    MixingParameters<float> mixingParameters(3, 2);

    mixingParameters.setGain(0, 0, 0);
    mixingParameters.setGain(1, 0, 20);
    mixingParameters.setGain(2, 0, 40);

    mixingParameters.setGain(0, 1, 40);
    mixingParameters.setGain(1, 1, 20);
    mixingParameters.setGain(2, 1, 0);

    EXPECT_EQ(mixingParameters.gains(), vector<float>({ 1, 10, 100, 100, 10, 1 }));
}
