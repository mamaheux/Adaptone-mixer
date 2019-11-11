#include <Utils/Time.h>

#include <gtest/gtest.h>

using namespace adaptone;

TEST(MathTests, waitUntilTimeReached_shouldWaitTheProperAmountOfTime)
{
    timespec ts;
    timespec_get(&ts, TIME_UTC);

    timespec thresholdTime = addMsToTimespec(1500, ts);

    waitUntilTimeReached(thresholdTime);

    timespec_get(&ts, TIME_UTC);

    EXPECT_EQ(ts.tv_sec - thresholdTime.tv_sec, 0);
    EXPECT_NEAR(ts.tv_nsec, thresholdTime.tv_nsec, 1000);
}

TEST(MathTests, addMsToTimespec_shouldAddAndNormalizeMillisecondeDelayToTimespecStructure)
{
    timespec baseTime;

    baseTime.tv_sec = 10;
    baseTime.tv_nsec = 999999999;

    timespec newTime = addMsToTimespec(1234, baseTime);

    EXPECT_EQ(newTime.tv_sec, 12);
    EXPECT_EQ(newTime.tv_nsec, 233999999);
}
