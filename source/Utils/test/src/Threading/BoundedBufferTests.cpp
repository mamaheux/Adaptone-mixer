#include <Utils/Threading/BoundedBuffer.h>

#include <gtest/gtest.h>

#include <atomic>
#include <string>
#include <vector>
#include <thread>

using namespace adaptone;
using namespace std;
using namespace std::chrono_literals;

TEST(BoundedBufferTests, constructor_count_shouldInitUsingTheDefaultConstructor)
{
    constexpr size_t BufferSize = 2;
    BoundedBuffer<string> boundedBuffer(BufferSize);

    for (size_t i = 0; i < BufferSize; i++)
    {
        boundedBuffer.write([](string& o)
        {
            EXPECT_EQ(o, "");
        });
    }
}

TEST(BoundedBufferTests, constructor_countInitializer_shouldInitUsingTheDefaultConstructor)
{
    constexpr size_t BufferSize = 2;
    BoundedBuffer<string> boundedBuffer(BufferSize, []()
    { return "a"; });

    for (size_t i = 0; i < BufferSize; i++)
    {
        boundedBuffer.write([](string& o)
        {
            EXPECT_EQ(o, "a");
        });
    }
}

TEST(BoundedBufferTests, constructor_countInitializerDeleter_shouldInitUsingTheDefaultConstructor)
{
    constexpr size_t BufferSize = 2;
    vector<size_t> deletedBuffers;

    {
        BoundedBuffer<size_t> boundedBuffer(BufferSize,
            []()
            { return 1; },
            [&](size_t& o)
            { deletedBuffers.push_back(o); });

        for (size_t i = 0; i < BufferSize; i++)
        {
            boundedBuffer.write([=](size_t& o)
            {
                EXPECT_EQ(o, 1);
                o = i;
            });
        }
    }

    EXPECT_EQ(deletedBuffers, vector<size_t>({ 0, 1 }));
}

TEST(BoundedBufferTests, producerConsumer_shouldHandleMultithreading)
{
    constexpr size_t BufferSize = 5;
    constexpr size_t ProducedMessageCount = 20;
    BoundedBuffer<size_t> boundedBuffer(BufferSize);

    thread producerThread([&]()
    {
        for (size_t i = 0; i < ProducedMessageCount; i++)
        {
            boundedBuffer.write([=](size_t& o)
            {
                o = i;
            });
        }
    });

    thread consumerThread([&]()
    {
        for (size_t i = 0; i < ProducedMessageCount; i++)
        {
            boundedBuffer.read([=](const size_t& o)
            {
                EXPECT_EQ(o, i);
            });

            this_thread::sleep_for(1ms);
        }
    });

    producerThread.join();
    consumerThread.join();
}

#define DECLARE_PERFORMANCE_TEST(BoundedBufferType) \
    TEST(BoundedBufferTests, producerConsumer_##BoundedBufferType##_performance) \
    { \
        constexpr size_t BufferSize = 5; \
        constexpr size_t ProducedMessageCount = 100000; \
        BoundedBufferType<size_t> boundedBuffer(BufferSize); \
     \
        thread producerThread([&]() \
        { \
            for (size_t i = 0; i < ProducedMessageCount; i++) \
            { \
                boundedBuffer.write([=](size_t& o) \
                { \
                    o = i; \
                }); \
            } \
        }); \
     \
        thread consumerThread([&]() \
        { \
            for (size_t i = 0; i < ProducedMessageCount; i++) \
            { \
                boundedBuffer.read([=](const size_t& o) \
                { \
                    EXPECT_EQ(o, i); \
                }); \
            } \
        }); \
     \
        auto start = chrono::system_clock::now(); \
        producerThread.join(); \
        consumerThread.join(); \
        auto end = chrono::system_clock::now(); \
        chrono::duration<double> elapsedSeconds = end - start; \
        cout << "Elapsed time (avg) = " << elapsedSeconds.count() / ProducedMessageCount << " s" << endl; \
    }

DECLARE_PERFORMANCE_TEST(BoundedBuffer)
DECLARE_PERFORMANCE_TEST(SpinLockBoundedBuffer)
