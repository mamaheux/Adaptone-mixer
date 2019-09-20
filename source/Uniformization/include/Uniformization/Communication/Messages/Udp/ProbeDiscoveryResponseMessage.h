#ifndef UNIFORMIZATION_COMMUNICATION_MESSAGES_UDP_PROBE_DISCOVERY_RESPONSE_MESSAGE_H
#define UNIFORMIZATION_COMMUNICATION_MESSAGES_UDP_PROBE_DISCOVERY_RESPONSE_MESSAGE_H

#include <Uniformization/Communication/Messages/ProbeMessage.h>

namespace adaptone
{
    class ProbeDiscoveryResponseMessage : public ProbeMessage
    {
    public:
        static constexpr uint32_t Id = 1;

        ProbeDiscoveryResponseMessage();
        ~ProbeDiscoveryResponseMessage() override;

    protected:
        void serialize(NetworkBufferView& buffer) override;
    };
}

#endif
