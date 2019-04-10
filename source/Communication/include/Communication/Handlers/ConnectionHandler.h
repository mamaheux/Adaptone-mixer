#ifndef COMMUNICATION_HANDLERS_CONNECTION_HANDLER_H
#define COMMUNICATION_HANDLERS_CONNECTION_HANDLER_H

#include <Utils/ClassMacro.h>

namespace adaptone
{
    class ConnectionHandler
    {
    public:
        ConnectionHandler();
        virtual ~ConnectionHandler();

        DECLARE_NOT_COPYABLE(ConnectionHandler);
        DECLARE_NOT_MOVABLE(ConnectionHandler);

        virtual void handleConnection() = 0;
        virtual void handleDisconnection() = 0;
    };
}

#endif
