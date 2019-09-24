#ifndef UNIFORMIZATION_COMMUNICATION_MESSAGES_TCP_RECORD_RESPONSE_MESSAGE_H
#define UNIFORMIZATION_COMMUNICATION_MESSAGES_TCP_RECORD_RESPONSE_MESSAGE_H

#include <Uniformization/Communication/Messages/PayloadMessage.h>

namespace adaptone
{
    /**
     * The data is not copied. So, the message life time depends on the network buffer life time.
     */
    class RecordResponseMessage : public PayloadMessage
    {
    public:
        static constexpr uint32_t Id = 6;
        static constexpr std::size_t MinimumMessageSize = 9;

    private:
        uint8_t m_recordId;
        const uint8_t* m_data;
        std::size_t m_dataSize;

    public:
        RecordResponseMessage(uint8_t recordId, const uint8_t* data, std::size_t dataSize);
        ~RecordResponseMessage() override;

        uint8_t recordId() const;
        const uint8_t* data() const;
        std::size_t dataSize() const;

        static RecordResponseMessage fromBuffer(NetworkBufferView buffer, std::size_t messageSize);

    protected:
        void serializePayload(NetworkBufferView buffer) const override;
    };

    inline uint8_t RecordResponseMessage::recordId() const
    {
        return m_recordId;
    }

    inline const uint8_t* RecordResponseMessage::data() const
    {
        return m_data;
    }

    inline std::size_t RecordResponseMessage::dataSize() const
    {
        return m_dataSize;
    }
}

#endif
