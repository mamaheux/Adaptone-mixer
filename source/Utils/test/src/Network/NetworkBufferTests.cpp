#include <Utils/Network/NetworkBuffer.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

TEST(NetworkBufferTests, constructor_shouldAllocateMemory)
{
    constexpr size_t Size = 1000;
    NetworkBuffer buffer(Size);

    ASSERT_EQ(buffer.size(), Size);
    for (size_t i = 0; i < Size; i++)
    {
        uint8_t v = static_cast<uint8_t>(i);
        buffer.data()[i] = v;
        EXPECT_EQ(buffer.data()[i], v);
    }
}

TEST(NetworkBufferTests, view_shouldReturnTheSpecifiedView)
{
    constexpr size_t Size = 4;
    NetworkBuffer buffer(Size);

    buffer.data()[0] = 1;
    buffer.data()[1] = 2;
    buffer.data()[2] = 3;
    buffer.data()[3] = 4;

    NetworkBufferView view0 = buffer.view(1);
    NetworkBufferView view1 = view0.view(1);

    ASSERT_EQ(view0.size(), Size - 1);
    EXPECT_EQ(view0.data()[0], 2);
    EXPECT_EQ(view0.data()[1], 3);
    EXPECT_EQ(view0.data()[2], 4);

    ASSERT_EQ(view1.size(), Size - 2);
    EXPECT_EQ(view1.data()[0], 3);
    EXPECT_EQ(view1.data()[1], 4);
}
