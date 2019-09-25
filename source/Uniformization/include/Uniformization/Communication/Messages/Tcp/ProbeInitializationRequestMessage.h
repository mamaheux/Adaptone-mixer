#ifndef UNIFORMIZATION_COMMUNICATION_MESSAGES_TCP_PROBE_INITIALIZATION_REQUEST_MESSAGE_H
#define UNIFORMIZATION_COMMUNICATION_MESSAGES_TCP_PROBE_INITIALIZATION_REQUEST_MESSAGE_H

#include <Uniformization/Communication/Messages/PayloadMessage.h>

#include <Utils/Data/PcmAudioFrameFormat.h>

namespace adaptone
{
    class ProbeInitializationRequestMessage : public PayloadMessage
    {
    public:
        static constexpr uint32_t Id = 2;
        static constexpr std::size_t MessageSize = 16;

    private:
        uint32_t m_sampleFrequency;
        PcmAudioFrameFormat m_format;

    public:
        ProbeInitializationRequestMessage(uint32_t sampleFrequency, PcmAudioFrameFormat format);
        ~ProbeInitializationRequestMessage() override;

        uint32_t sampleFrequency() const;
        PcmAudioFrameFormat format() const;

        static ProbeInitializationRequestMessage fromBuffer(NetworkBufferView buffer, std::size_t messageSize);

    protected:
        void serializePayload(NetworkBufferView buffer) const override;

        static uint32_t serializeFormat(PcmAudioFrameFormat format);
        static PcmAudioFrameFormat parseFormat(uint32_t format);
    };

    inline uint32_t ProbeInitializationRequestMessage::sampleFrequency() const
    {
        return m_sampleFrequency;
    }

    inline PcmAudioFrameFormat ProbeInitializationRequestMessage::format() const
    {
        return m_format;
    }
}

#endif
