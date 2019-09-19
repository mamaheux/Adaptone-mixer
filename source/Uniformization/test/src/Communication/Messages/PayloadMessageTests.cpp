#include <Uniformization/Communication/Messages/PayloadMessage.h>

#include <gtest/gtest.h>

using namespace adaptone;
using namespace std;

class DummyPayloadMessage : public PayloadMessage
{

public:
    DummyPayloadMessage() : PayloadMessage(1, 2)
    {
    }

    ~DummyPayloadMessage() override
    {
    }

protected:
    void serializePayload(NetworkBufferView& buffer) override
    {
        buffer.data()[0] = 10;
        buffer.data()[1] = 11;
    }
};

TEST(PayloadMessageTests, toBuffer_shouldSerializeTheMessage)
{
    DummyPayloadMessage message;
    NetworkBuffer buffer(10);

    message.toBuffer(buffer);

    EXPECT_EQ(message.id(), 1);
    EXPECT_EQ(message.fullSize(), 10);
    EXPECT_EQ(message.payloadSize(), 2);

    EXPECT_EQ(buffer.data()[0], 0);
    EXPECT_EQ(buffer.data()[1], 0);
    EXPECT_EQ(buffer.data()[2], 0);
    EXPECT_EQ(buffer.data()[3], 1);
    EXPECT_EQ(buffer.data()[4], 0);
    EXPECT_EQ(buffer.data()[5], 0);
    EXPECT_EQ(buffer.data()[6], 0);
    EXPECT_EQ(buffer.data()[7], 2);

    EXPECT_EQ(buffer.data()[8], 10);
    EXPECT_EQ(buffer.data()[9], 11);
}
