#ifndef UNIFORMIZATION_COMMUNICATION_MESSAGES_TCP_FFT_REQUEST_MESSAGE_H
#define UNIFORMIZATION_COMMUNICATION_MESSAGES_TCP_FFT_REQUEST_MESSAGE_H

#include <Uniformization/Communication/Messages/PayloadMessage.h>

namespace adaptone
{
    class FftRequestMessage : public PayloadMessage
    {
    public:
        static constexpr uint32_t Id = 7;

    private:
        uint8_t m_hours;
        uint8_t m_minutes;
        uint8_t m_seconds;
        uint16_t m_milliseconds;
        uint16_t m_fftId;

    public:
        FftRequestMessage(uint8_t hours, uint8_t minutes, uint8_t seconds, uint16_t milliseconds, uint16_t fftId);
        ~FftRequestMessage() override;

        uint8_t hours() const;
        uint8_t minutes() const;
        uint8_t seconds() const;
        uint16_t milliseconds() const;
        uint16_t fftId() const;

        static FftRequestMessage fromBuffer(NetworkBufferView buffer, std::size_t messageSize);

    protected:
        void serializePayload(NetworkBufferView buffer) const override;
    };

    inline uint8_t FftRequestMessage::hours() const
    {
        return m_hours;
    }

    inline uint8_t FftRequestMessage::minutes() const
    {
        return m_minutes;
    }

    inline uint8_t FftRequestMessage::seconds() const
    {
        return m_seconds;
    }

    inline uint16_t FftRequestMessage::milliseconds() const
    {
        return m_milliseconds;
    }

    inline uint16_t FftRequestMessage::fftId() const
    {
        return m_fftId;
    }
}

#endif
