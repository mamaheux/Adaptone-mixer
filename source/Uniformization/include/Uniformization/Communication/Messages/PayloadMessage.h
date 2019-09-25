#ifndef UNIFORMIZATION_COMMUNICATION_MESSAGES_PAYLOAD_MESSAGE_H
#define UNIFORMIZATION_COMMUNICATION_MESSAGES_PAYLOAD_MESSAGE_H

#include <Uniformization/Communication/Messages/ProbeMessage.h>

namespace adaptone
{
    class PayloadMessage : public ProbeMessage
    {
        uint32_t m_payloadSize;

    public:
        PayloadMessage(uint32_t id, std::size_t payloadSize);
        ~PayloadMessage() override;

        uint32_t payloadSize();

    protected:
        void serialize(NetworkBufferView buffer) const override;
        virtual void serializePayload(NetworkBufferView buffer) const = 0;
    };

    inline uint32_t PayloadMessage::payloadSize()
    {
        return m_payloadSize;
    }
}

#endif
