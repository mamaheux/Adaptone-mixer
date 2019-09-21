#ifndef UNIFORMIZATION_COMMUNICATION_MESSAGES_UDP_PROBE_DISCOVERY_REQUEST_MESSAGE_H
#define UNIFORMIZATION_COMMUNICATION_MESSAGES_UDP_PROBE_DISCOVERY_REQUEST_MESSAGE_H

#include <Uniformization/Communication/Messages/ProbeMessage.h>

namespace adaptone
{
    class ProbeDiscoveryRequestMessage : public ProbeMessage
    {
    public:
        static constexpr uint32_t Id = 0;

        ProbeDiscoveryRequestMessage();
        ~ProbeDiscoveryRequestMessage() override;

        static ProbeDiscoveryRequestMessage fromBuffer(NetworkBufferView buffer, std::size_t messageSize);

    protected:
        void serialize(NetworkBufferView buffer) const override;
    };
}

#endif
