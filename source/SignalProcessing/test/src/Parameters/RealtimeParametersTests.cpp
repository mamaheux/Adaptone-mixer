#include <SignalProcessing/Parameters/RealtimeParameters.h>

#include <gtest/gtest.h>

#include <thread>

using namespace adaptone;
using namespace std;

TEST(RealtimeParametersTests, update_shouldExecuteTheFunctionAndSetDirtyToTrue)
{
    RealtimeParameters realtimeParameters;

    int counter = 0;
    realtimeParameters.uptate([&]()
    {
        counter++;
    });

    EXPECT_EQ(counter, 1);
    EXPECT_TRUE(realtimeParameters.isDirty());
}

TEST(RealtimeParametersTests, applyUpdate_shouldExecuteTheFunctionOnlyIfTheParametersAreDirtyAndSetDirtyToFalse)
{
    RealtimeParameters realtimeParameters;

    int counter = 0;
    auto function = [&]()
    {
        counter++;
    };

    realtimeParameters.applyUptate(function);

    EXPECT_EQ(counter, 0);
    EXPECT_FALSE(realtimeParameters.isDirty());

    realtimeParameters.uptate([&]()
    {
    });

    realtimeParameters.applyUptate(function);

    EXPECT_EQ(counter, 1);
    EXPECT_FALSE(realtimeParameters.isDirty());
}

TEST(RealtimeParametersTests, tryApplyingUpdate_shouldExecuteTheFunctionOnlyIfTheParametersAreDirtyAndSetDirtyToFalse)
{
    RealtimeParameters realtimeParameters;

    int counter = 0;
    auto function = [&]()
    {
        counter++;
    };

    EXPECT_TRUE(realtimeParameters.tryApplyingUptate(function));
    EXPECT_EQ(counter, 0);
    EXPECT_FALSE(realtimeParameters.isDirty());

    thread t([&]()
    {
        realtimeParameters.uptate([&]()
        {
            this_thread::sleep_for(0.2s);
        });
    });

    this_thread::sleep_for(0.1s);
    EXPECT_FALSE(realtimeParameters.tryApplyingUptate(function));
    EXPECT_EQ(counter, 0);

    t.join();
    EXPECT_TRUE(realtimeParameters.isDirty());

    EXPECT_TRUE(realtimeParameters.tryApplyingUptate(function));
    EXPECT_EQ(counter, 1);
    EXPECT_FALSE(realtimeParameters.isDirty());
}
