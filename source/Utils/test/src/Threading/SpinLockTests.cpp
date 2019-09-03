#include <Utils/Threading/SpinLock.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <thread>
#include <chrono>

using namespace adaptone;
using namespace std;
using namespace std::chrono_literals;


TEST(SpinLockTests, spinLock_shouldProtectResources)
{
    atomic_bool isFinished(false);
    atomic_int assignationCount(0);
    SpinLock s0, s1, s2;
    volatile int value = -1;

    s0.lock();
    s1.lock();
    s2.lock();

    std::thread t0([&]()
    {
        while (!isFinished.load())
        {
            s0.lock();
            value = 0;
            assignationCount++;
        }
    });

    std::thread t1([&]()
    {
        while (!isFinished.load())
        {
            s1.lock();
            value = 1;
            assignationCount++;
        }
    });

    std::thread t2([&]()
    {
        while (!isFinished.load())
        {
            s2.lock();
            value = 2;
            assignationCount++;
        }
    });

    EXPECT_EQ(value, -1);

    for (int i = 0; i < 2; i++)
    {
        s0.unlock();
        this_thread::sleep_for(100ms);
        EXPECT_EQ(value, 0);

        s2.unlock();
        this_thread::sleep_for(100ms);
        EXPECT_EQ(value, 2);

        s1.unlock();
        this_thread::sleep_for(100ms);
        EXPECT_EQ(value, 1);
    }

    EXPECT_EQ(assignationCount.load(), 6);

    isFinished.store(true);
    s0.unlock();
    s1.unlock();
    s2.unlock();
    t0.join();
    t1.join();
    t2.join();
}
