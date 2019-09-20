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

    protected:
        void serialize(NetworkBufferView buffer) override;
    };
}

#endif
