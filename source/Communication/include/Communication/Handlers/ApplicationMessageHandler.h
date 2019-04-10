#ifndef COMMUNICATION_HANDLERS_APPLICATION_MESSAGE_HANDLER_H
#define COMMUNICATION_HANDLERS_APPLICATION_MESSAGE_HANDLER_H

#include <Communication/Messages/ApplicationMessage.h>

#include <Utils/ClassMacro.h>

#include <nlohmann/json.hpp>

namespace adaptone
{
    class ApplicationMessageHandler
    {
    public:
        ApplicationMessageHandler();
        virtual ~ApplicationMessageHandler();

        DECLARE_NOT_COPYABLE(ApplicationMessageHandler);
        DECLARE_NOT_MOVABLE(ApplicationMessageHandler);

        void handle(nlohmann::json& j);

    protected:
        virtual void handle(const ApplicationMessage& message) = 0;
    };
}

#endif
