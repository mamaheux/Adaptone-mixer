#include <Utils/Functional/FunctionQueue.h>

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

using namespace adaptone;
using namespace std;

TEST(FunctionQueueTests, boolReturnType_shouldRemoveTheFunctionOnlyIfItReturnsTrue)
{
    int counter1 = 0;
    int counter2 = 0;

    FunctionQueue<bool()> functionQueue;
    EXPECT_EQ(functionQueue.size(), 0);

    functionQueue.push([&]()
    {
        counter1++;
        return counter1 != 1;
    });
    EXPECT_EQ(functionQueue.size(), 1);

    functionQueue.push([&]()
    {
        counter2++;
        return true;
    });
    EXPECT_EQ(functionQueue.size(), 2);

    functionQueue.execute();
    EXPECT_EQ(functionQueue.size(), 2);

    functionQueue.execute();
    EXPECT_EQ(functionQueue.size(), 1);

    functionQueue.execute();
    EXPECT_EQ(functionQueue.size(), 0);

    functionQueue.execute(); //this call should do nothing

    EXPECT_EQ(counter1, 2);
    EXPECT_EQ(counter2, 1);
}

TEST(FunctionQueueTests, voidReturnType_shouldRemoveTheFunction)
{
    int counter1 = 0;
    int counter2 = 0;

    FunctionQueue<void()> functionQueue;
    EXPECT_EQ(functionQueue.size(), 0);

    functionQueue.push([&]()
    {
        counter1++;
    });
    EXPECT_EQ(functionQueue.size(), 1);

    functionQueue.push([&]()
    {
        counter2++;
    });
    EXPECT_EQ(functionQueue.size(), 2);

    functionQueue.execute();
    EXPECT_EQ(functionQueue.size(), 1);

    functionQueue.execute();
    EXPECT_EQ(functionQueue.size(), 0);

    functionQueue.execute(); //this call should do nothing

    EXPECT_EQ(counter1, 1);
    EXPECT_EQ(counter2, 1);
}

TEST(FunctionQueueTests, multithread_shouldPushAndExecuteWithoutError)
{
    constexpr int CounterMax = 100;
    int counter = 0;

    FunctionQueue<void()> functionQueue;

    thread t([&]()
    {
        this_thread::sleep_for(0.001s);
        for (int i = 0; i < CounterMax; i++)
        {
            functionQueue.push([&]()
            {
                counter++;
            });
        }
    });

    auto start = chrono::system_clock::now();
    while (counter < CounterMax)
    {
        if (chrono::system_clock::now() - start > 1s)
        {
            t.join();
            FAIL();
        }
        functionQueue.tryExecute();
    }

    t.join();
    SUCCEED();
}
