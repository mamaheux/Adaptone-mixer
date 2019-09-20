#ifndef UNIFORMIZATION_COMMUNICATION_MESSAGES_TCP_RECORD_REQUEST_MESSAGE_H
#define UNIFORMIZATION_COMMUNICATION_MESSAGES_TCP_RECORD_REQUEST_MESSAGE_H

#include <Uniformization/Communication/Messages/PayloadMessage.h>

namespace adaptone
{
    class RecordRequestMessage : public PayloadMessage
    {
    public:
        static constexpr uint32_t Id = 5;

    private:
        uint8_t m_hours;
        uint8_t m_minutes;
        uint8_t m_seconds;
        uint16_t m_milliseconds;
        uint16_t m_duration;
        uint8_t m_recordId;

    public:
        RecordRequestMessage(uint8_t hours, uint8_t minutes, uint8_t seconds, uint16_t milliseconds, uint16_t duration,
            uint8_t recordId);
        ~RecordRequestMessage() override;

        uint8_t hours() const;
        uint8_t minutes() const;
        uint8_t seconds() const;
        uint16_t milliseconds() const;
        uint16_t duration() const;
        uint8_t recordId() const;

        static RecordRequestMessage fromBuffer(NetworkBufferView buffer, size_t messageSize);

    protected:
        void serializePayload(NetworkBufferView buffer) override;
    };

    inline uint8_t RecordRequestMessage::hours() const
    {
        return m_hours;
    }

    inline uint8_t RecordRequestMessage::minutes() const
    {
        return m_minutes;
    }

    inline uint8_t RecordRequestMessage::seconds() const
    {
        return m_seconds;
    }

    inline uint16_t RecordRequestMessage::milliseconds() const
    {
        return m_milliseconds;
    }

    inline uint16_t RecordRequestMessage::duration() const
    {
        return m_duration;
    }

    inline uint8_t RecordRequestMessage::recordId() const
    {
        return m_recordId;
    }
}

#endif
