#ifndef UNIFORMIZATION_COMMUNICATION_PROBE_MESSAGE_HANDLER_H
#define UNIFORMIZATION_COMMUNICATION_PROBE_MESSAGE_HANDLER_H

#include <Uniformization/Communication/Messages/ProbeMessage.h>

#include <Utils/ClassMacro.h>

namespace adaptone
{
    class ProbeMessageHandler
    {
    public:
        ProbeMessageHandler();
        virtual ~ProbeMessageHandler();

        DECLARE_NOT_COPYABLE(ProbeMessageHandler);
        DECLARE_NOT_MOVABLE(ProbeMessageHandler);

        virtual void handle(const ProbeMessage& message, std::size_t probeId, bool isMaster) = 0;
    };
}

#endif
