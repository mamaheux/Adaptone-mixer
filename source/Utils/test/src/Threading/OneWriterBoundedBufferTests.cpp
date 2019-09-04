#include <Utils/Threading/OneWriterBoundedBuffer.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <atomic>
#include <string>
#include <vector>
#include <thread>

using namespace adaptone;
using namespace std;
using namespace std::chrono_literals;

TEST(OneWriterBoundedBufferTests, producerConsumer_shouldHandleMultithreading)
{
    constexpr size_t BufferSize = 5;
    constexpr size_t ProducedMessageCount = 20;
    OneWriterBoundedBuffer<size_t> boundedBuffer(BufferSize);

    thread producerThread([&]()
    {
        for (size_t i = 0; i < ProducedMessageCount; i++)
        {
            boundedBuffer.writePartialData([=](size_t& o)
            {
                o = i;
            });
            this_thread::sleep_for(1ms);
            boundedBuffer.writePartialData([=](size_t& o)
            {
                o += 100;
            });
            boundedBuffer.finishWriting();
        }
    });

    thread consumerThread([&]()
    {
        for (size_t i = 0; i < ProducedMessageCount; i++)
        {
            boundedBuffer.read([=](const size_t& o)
            {
                EXPECT_EQ(o, i + 100);
            });

            this_thread::sleep_for(2ms);
        }
    });

    producerThread.join();
    consumerThread.join();
}
