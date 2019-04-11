#ifndef COMMUNICATION_MESAGES_INITIALIZATION_LAUNCH_INITILIZATION_MESSAGE_H
#define COMMUNICATION_MESAGES_INITIALIZATION_LAUNCH_INITILIZATION_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

namespace adaptone
{
    class LaunchInitializationMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 2;

        LaunchInitializationMessage();
        virtual ~LaunchInitializationMessage();

        friend void to_json(nlohmann::json& j, const LaunchInitializationMessage& o);
        friend void from_json(const nlohmann::json& j, LaunchInitializationMessage& o);
    };

    inline void to_json(nlohmann::json& j, const LaunchInitializationMessage& o)
    {
        j = nlohmann::json{{ "seqId", o.seqId() }};
    }

    inline void from_json(const nlohmann::json& j, LaunchInitializationMessage& o)
    {
    }
}

#endif
