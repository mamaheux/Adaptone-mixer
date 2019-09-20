#ifndef UNIFORMIZATION_COMMUNICATION_MESSAGES_TCP_HEARTBEAT_MESSAGE_H
#define UNIFORMIZATION_COMMUNICATION_MESSAGES_TCP_HEARTBEAT_MESSAGE_H

#include <Uniformization/Communication/Messages/ProbeMessage.h>

namespace adaptone
{
    class HeartbeatMessage : public ProbeMessage
    {
    public:
        static constexpr uint32_t Id = 4;

        HeartbeatMessage();
        ~HeartbeatMessage() override;

    protected:
        void serialize(NetworkBufferView& buffer) override;
    };
}

#endif
