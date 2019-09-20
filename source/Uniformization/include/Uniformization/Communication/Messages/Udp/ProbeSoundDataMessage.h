#ifndef UNIFORMIZATION_COMMUNICATION_MESSAGES_UDP_PROBE_SOUND_DATA_MESSAGE_H
#define UNIFORMIZATION_COMMUNICATION_MESSAGES_UDP_PROBE_SOUND_DATA_MESSAGE_H

#include <Uniformization/Communication/Messages/PayloadMessage.h>

namespace adaptone
{
    /**
     * The data is not copied. So, the message life time depends on the network buffer life time.
     */
    class ProbeSoundDataMessage : public PayloadMessage
    {
    public:
        static constexpr uint32_t Id = 9;

    private:
        uint16_t m_soundDataId;
        uint8_t m_hours;
        uint8_t m_minutes;
        uint8_t m_seconds;
        uint16_t m_milliseconds;
        uint16_t m_microseconds;
        const uint8_t* m_data;
        std::size_t m_dataSize;

    public:
        ProbeSoundDataMessage(uint16_t soundDataId, uint8_t hours, uint8_t minutes, uint8_t seconds,
            uint16_t milliseconds, uint16_t microseconds, const uint8_t* data, std::size_t dataSize);
        ~ProbeSoundDataMessage() override;

        uint16_t soundDataId() const;
        uint8_t hours() const;
        uint8_t minutes() const;
        uint8_t seconds() const;
        uint16_t milliseconds() const;
        uint16_t microseconds() const;
        const uint8_t* data() const;
        std::size_t dataSize() const;

        static ProbeSoundDataMessage fromBuffer(NetworkBufferView buffer, size_t messageSize);

    protected:
        void serializePayload(NetworkBufferView buffer) override;
    };

    inline uint16_t ProbeSoundDataMessage::soundDataId() const
    {
        return m_soundDataId;
    }

    inline uint8_t ProbeSoundDataMessage::hours() const
    {
        return m_hours;
    }

    inline uint8_t ProbeSoundDataMessage::minutes() const
    {
        return m_minutes;
    }

    inline uint8_t ProbeSoundDataMessage::seconds() const
    {
        return m_seconds;
    }

    inline uint16_t ProbeSoundDataMessage::milliseconds() const
    {
        return m_milliseconds;
    }

    inline uint16_t ProbeSoundDataMessage::microseconds() const
    {
        return m_microseconds;
    }

    inline const uint8_t* ProbeSoundDataMessage::data() const
    {
        return m_data;
    }

    inline std::size_t ProbeSoundDataMessage::dataSize() const
    {
        return m_dataSize;
    }
}

#endif
