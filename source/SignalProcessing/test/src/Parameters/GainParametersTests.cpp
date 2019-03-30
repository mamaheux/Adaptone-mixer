#include <SignalProcessing/Parameters/GainParameters.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(GainParametersTests, constructor_shouldInitializeAllGainsTo1)
{
    GainParameters<float> gainParameters(2);

    EXPECT_EQ(gainParameters.gains(), vector<float>({ 1, 1 }));
}

TEST(GainParametersTests, setGain_invalidChannel_shouldThrowInvalidException)
{
    GainParameters<float> gainParameters(2);

    EXPECT_THROW(gainParameters.setGain(2, 20), InvalidValueException);
}

TEST(GainParametersTests, setGain_shouldSetSpecifiedChannelGainNotInDb)
{
    GainParameters<float> gainParameters(2);

    EXPECT_FALSE(gainParameters.isDirty());

    gainParameters.setGain(0, 20);

    EXPECT_TRUE(gainParameters.isDirty());
    EXPECT_EQ(gainParameters.gains(), vector<float>({ 10, 1 }));
}

TEST(GainParametersTests, setGains_invalidChannelCount_shouldThrowInvalidException)
{
    GainParameters<float> gainParameters(2);
    EXPECT_THROW(gainParameters.setGains({ 1, 2, 3 }), InvalidValueException);
}

TEST(GainParametersTests, setGains_shouldSetAllGainsNotInDb)
{
    GainParameters<float> gainParameters(2);

    EXPECT_FALSE(gainParameters.isDirty());

    gainParameters.setGains({ 20, 40 });

    EXPECT_TRUE(gainParameters.isDirty());
    EXPECT_EQ(gainParameters.gains(), vector<float>({ 10, 100 }));
}
