#ifndef COMMUNICATION_HANDLERS_APPLICATION_MESSAGE_HANDLER_H
#define COMMUNICATION_HANDLERS_APPLICATION_MESSAGE_HANDLER_H

#include <Communication/Messages/ApplicationMessage.h>

#include <Utils/ClassMacro.h>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <unordered_map>
#include <functional>

namespace adaptone
{
    class ApplicationMessageHandler
    {
        std::unordered_map<std::size_t, std::function<void(nlohmann::json&)>> m_handleFunctions;

    public:
        ApplicationMessageHandler();
        virtual ~ApplicationMessageHandler();

        DECLARE_NOT_COPYABLE(ApplicationMessageHandler);
        DECLARE_NOT_MOVABLE(ApplicationMessageHandler);

        void handle(nlohmann::json& j);

    protected:
        virtual void handleDeserialized(const ApplicationMessage& message) = 0;
    };
}

#endif
