#ifndef COMMUNICATION_MESAGES_INITIALIZATION_RELAUNCH_INITILIZATION_MESSAGE_H
#define COMMUNICATION_MESAGES_INITIALIZATION_RELAUNCH_INITILIZATION_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

#include <cstddef>
#include <string>

namespace adaptone
{
    class RelaunchInitializationMessage : public ApplicationMessage
    {

    public:
        RelaunchInitializationMessage();
        virtual ~RelaunchInitializationMessage();

        friend void to_json(nlohmann::json& j, const RelaunchInitializationMessage& o);
        friend void from_json(const nlohmann::json& j, RelaunchInitializationMessage& o);
    };

    inline void to_json(nlohmann::json& j, const RelaunchInitializationMessage& o)
    {
        j = nlohmann::json{{ "seqId", o.seqId() }};
    }

    inline void from_json(const nlohmann::json& j, RelaunchInitializationMessage& o)
    {
    }
}

#endif
