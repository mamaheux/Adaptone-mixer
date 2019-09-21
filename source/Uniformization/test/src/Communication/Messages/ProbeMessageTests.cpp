#include <Uniformization/Communication/Messages/ProbeMessage.h>

#include <Utils/Exception/InvalidValueException.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

class DummyMessage : public ProbeMessage
{

public:
    DummyMessage() : ProbeMessage(1, 1)
    {
    }

    ~DummyMessage() override
    {
    }

protected:
    void serialize(NetworkBufferView buffer) const override
    {
        buffer.data()[0] = 10;
    }
};

TEST(ProbeMessageTests, toBuffer_tooSmallBuffer_shouldThrowInvalidValueException)
{
    DummyMessage message;
    NetworkBuffer buffer(1);

    EXPECT_THROW(message.toBuffer(buffer), InvalidValueException);
}

TEST(ProbeMessageTests, toBuffer_shouldSerializeTheMessage)
{
    DummyMessage message;
    NetworkBuffer buffer(10);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 1);
    EXPECT_EQ(message.fullSize(), 5);

    EXPECT_EQ(buffer.data()[0], 0);
    EXPECT_EQ(buffer.data()[1], 0);
    EXPECT_EQ(buffer.data()[2], 0);
    EXPECT_EQ(buffer.data()[3], 1);
    EXPECT_EQ(buffer.data()[4], 10);
}

TEST(ProbeMessageTests, hasPayload_shouldReturnTrueIfTheMessageHasAPayload)
{
    EXPECT_FALSE(ProbeMessage::hasPayload(0));
    EXPECT_FALSE(ProbeMessage::hasPayload(1));
    EXPECT_TRUE(ProbeMessage::hasPayload(2));
    EXPECT_TRUE(ProbeMessage::hasPayload(3));
    EXPECT_FALSE(ProbeMessage::hasPayload(4));
    EXPECT_TRUE(ProbeMessage::hasPayload(5));
    EXPECT_TRUE(ProbeMessage::hasPayload(6));
    EXPECT_TRUE(ProbeMessage::hasPayload(7));
    EXPECT_TRUE(ProbeMessage::hasPayload(8));
    EXPECT_TRUE(ProbeMessage::hasPayload(9));
}

TEST(ProbeMessageTests, hasPayload_invalidId_shouldReturnTrueIfTheMessageHasAPayload)
{
    EXPECT_THROW(ProbeMessage::hasPayload(10), InvalidValueException);
}
