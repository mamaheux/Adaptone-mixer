#ifndef UNIFORMIZATION_COMMUNICATION_MESSAGES_UDP_PROBE_DISCOVERY_RESPONSE_MESSAGE_H
#define UNIFORMIZATION_COMMUNICATION_MESSAGES_UDP_PROBE_DISCOVERY_RESPONSE_MESSAGE_H

#include <Uniformization/Communication/Messages/ProbeMessage.h>

namespace adaptone
{
    class ProbeDiscoveryResponseMessage : public ProbeMessage
    {
    public:
        static constexpr uint32_t Id = 1;
        static constexpr std::size_t MessageSize = 4;

        ProbeDiscoveryResponseMessage();
        ~ProbeDiscoveryResponseMessage() override;

        static ProbeDiscoveryResponseMessage fromBuffer(NetworkBufferView buffer, std::size_t messageSize);

    protected:
        void serialize(NetworkBufferView buffer) const override;
    };
}

#endif
