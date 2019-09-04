#include <Utils/Threading/SpinLockSemaphore.h>

#include <gtest/gtest.h>

#include <thread>
#include <chrono>

using namespace adaptone;
using namespace std;
using namespace std::chrono_literals;


TEST(SpinLockSemaphoreTests, binarySpinLockSemaphore_shouldProtectResources)
{
    atomic_bool isFinished(false);
    atomic_int assignationCount(0);
    SpinLockSemaphore s0(0), s1(1), s2(0);
    volatile int value = -1;

    thread t0([&]()
    {
        while (!isFinished.load())
        {
            s0.wait();
            value = 0;
            assignationCount++;
        }
    });

    thread t1([&]()
    {
        while (!isFinished.load())
        {
            s1.wait();
            value = 1;
            assignationCount++;
        }
    });

    thread t2([&]()
    {
        while (!isFinished.load())
        {
            s2.wait();
            value = 2;
            assignationCount++;
        }
    });

    this_thread::sleep_for(100ms);
    EXPECT_EQ(value, 1);

    for (int i = 0; i < 2; i++)
    {
        s0.notify();
        this_thread::sleep_for(100ms);
        EXPECT_EQ(value, 0);

        s2.notify();
        this_thread::sleep_for(100ms);
        EXPECT_EQ(value, 2);

        s1.notify();
        this_thread::sleep_for(100ms);
        EXPECT_EQ(value, 1);
    }

    EXPECT_EQ(assignationCount.load(), 7);

    isFinished.store(true);
    s0.notify();
    s1.notify();
    s2.notify();
    t0.join();
    t1.join();
    t2.join();
}

TEST(SpinLockSemaphoreTests, ternarySpinLockSemaphore_shouldProtectResources)
{
    atomic_bool isFinished(false);
    SpinLockSemaphore s0(2);
    volatile int value = 0;

    thread t0([&]()
    {
        while (!isFinished.load())
        {
            s0.wait();
            value++;
        }
    });

    this_thread::sleep_for(100ms);
    EXPECT_EQ(value, 2);

    isFinished.store(true);
    s0.notify();
    t0.join();
}
