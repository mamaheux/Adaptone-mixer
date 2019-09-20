#ifndef UNIFORMIZATION_COMMUNICATION_MESSAGES_TCP_PROBE_INITIALIZATION_REQUEST_MESSAGE_H
#define UNIFORMIZATION_COMMUNICATION_MESSAGES_TCP_PROBE_INITIALIZATION_REQUEST_MESSAGE_H

#include <Uniformization/Communication/Messages/PayloadMessage.h>

#include <Utils/Data/PcmAudioFrame.h>

namespace adaptone
{
    class ProbeInitializationRequestMessage : public PayloadMessage
    {
    public:
        static constexpr uint32_t Id = 2;

    private:
        uint32_t m_sampleFrequency;
        PcmAudioFrame::Format m_format;

    public:
        ProbeInitializationRequestMessage(uint32_t sampleFrequency, PcmAudioFrame::Format format);
        ~ProbeInitializationRequestMessage() override;

        uint32_t sampleFrequency() const;
        PcmAudioFrame::Format format() const;

        static ProbeInitializationRequestMessage fromBuffer(NetworkBufferView buffer, size_t messageSize);

    protected:
        void serializePayload(NetworkBufferView buffer) override;

        static uint32_t serializeFormat(PcmAudioFrame::Format format);
        static PcmAudioFrame::Format parseFormat(uint32_t format);
    };

    inline uint32_t ProbeInitializationRequestMessage::sampleFrequency() const
    {
        return m_sampleFrequency;
    }

    inline PcmAudioFrame::Format ProbeInitializationRequestMessage::format() const
    {
        return m_format;
    }
}

#endif
