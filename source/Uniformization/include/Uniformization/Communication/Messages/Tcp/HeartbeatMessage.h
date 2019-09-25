#ifndef UNIFORMIZATION_COMMUNICATION_MESSAGES_TCP_HEARTBEAT_MESSAGE_H
#define UNIFORMIZATION_COMMUNICATION_MESSAGES_TCP_HEARTBEAT_MESSAGE_H

#include <Uniformization/Communication/Messages/ProbeMessage.h>

namespace adaptone
{
    class HeartbeatMessage : public ProbeMessage
    {
    public:
        static constexpr uint32_t Id = 4;
        static constexpr std::size_t MessageSize = 4;

        HeartbeatMessage();
        ~HeartbeatMessage() override;

        static HeartbeatMessage fromBuffer(NetworkBufferView buffer, std::size_t messageSize);

    protected:
        void serialize(NetworkBufferView buffer) const override;
    };
}

#endif
