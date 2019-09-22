#ifndef UNIFORMIZATION_COMMUNICATION_MESSAGES_TCP_PROBE_INITIALIZATION_RESPONSE_MESSAGE_H
#define UNIFORMIZATION_COMMUNICATION_MESSAGES_TCP_PROBE_INITIALIZATION_RESPONSE_MESSAGE_H

#include <Uniformization/Communication/Messages/PayloadMessage.h>

namespace adaptone
{
    class ProbeInitializationResponseMessage : public PayloadMessage
    {
    public:
        static constexpr uint32_t Id = 3;
        static constexpr std::size_t MessageSize = 10;

    private:
        bool m_isCompatible;
        bool m_isMaster;

    public:
        ProbeInitializationResponseMessage(bool isCompatible, bool isMaster);
        ~ProbeInitializationResponseMessage() override;

        bool isCompatible() const;
        bool isMaster() const;

        static ProbeInitializationResponseMessage fromBuffer(NetworkBufferView buffer, std::size_t messageSize);

    protected:
        void serializePayload(NetworkBufferView buffer) const override;
    };

    inline bool ProbeInitializationResponseMessage::isCompatible() const
    {
        return m_isCompatible;
    }

    inline bool ProbeInitializationResponseMessage::isMaster() const
    {
        return m_isMaster;
    }
}

#endif
