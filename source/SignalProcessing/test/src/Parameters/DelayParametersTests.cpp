#include <SignalProcessing/Parameters/DelayParameters.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

constexpr size_t MaxDelay = 4096;

TEST(DelayParametersTests, constructor_shouldInitializeAllDelaysTo0)
{
    DelayParameters delayParameters(2, MaxDelay);

    EXPECT_EQ(delayParameters.delays(), vector<size_t>({ 0, 0 }));
}

TEST(DelayParametersTests, setDelay_invalidChannel_shouldThrowInvalidException)
{
    DelayParameters delayParameters(2, MaxDelay);

    EXPECT_THROW(delayParameters.setDelay(2, 20), InvalidValueException);
}

TEST(DelayParametersTests, setDelay_invalidDelay_shouldThrowInvalidException)
{
    DelayParameters delayParameters(2, MaxDelay);

    EXPECT_THROW(delayParameters.setDelay(2, MaxDelay + 1), InvalidValueException);
}

TEST(DelayParametersTests, setDelay_shouldSetSpecifiedChannelDelay)
{
    DelayParameters delayParameters(2, MaxDelay);

    EXPECT_FALSE(delayParameters.isDirty());

    delayParameters.setDelay(0, 10);

    EXPECT_TRUE(delayParameters.isDirty());
    EXPECT_EQ(delayParameters.delays(), vector<size_t>({ 10, 0 }));
}

TEST(DelayParametersTests, setDelays_invalidChannelCount_shouldThrowInvalidException)
{
    DelayParameters delayParameters(2, MaxDelay);
    EXPECT_THROW(delayParameters.setDelays({ 1, 2, 3 }), InvalidValueException);
}

TEST(DelayParametersTests, setDelays_invalidDelay_shouldThrowInvalidException)
{
    DelayParameters delayParameters(2, MaxDelay);
    EXPECT_THROW(delayParameters.setDelays({ 1, MaxDelay + 1 }), InvalidValueException);
}

TEST(DelayParametersTests, setDelays_shouldSetAllDelays)
{
    DelayParameters delayParameters(2, MaxDelay);

    EXPECT_FALSE(delayParameters.isDirty());

    delayParameters.setDelays({ 10, 100 });

    EXPECT_TRUE(delayParameters.isDirty());
    EXPECT_EQ(delayParameters.delays(), vector<size_t>({ 10, 100 }));
}
