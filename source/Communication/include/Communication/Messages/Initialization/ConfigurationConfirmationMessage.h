#ifndef COMMUNICATION_MESAGES_INITIALIZATION_OPTIMIZE_POSITION_MESSAGE_H
#define COMMUNICATION_MESAGES_INITIALIZATION_OPTIMIZE_POSITION_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

namespace adaptone
{
    class ConfigurationConfirmationMessage : public ApplicationMessage
    {
    public:
        ConfigurationConfirmationMessage();
        virtual ~ConfigurationConfirmationMessage();

        friend void to_json(nlohmann::json& j, const ConfigurationConfirmationMessage& o);
        friend void from_json(const nlohmann::json& j, ConfigurationConfirmationMessage& o);
    };

    inline void to_json(nlohmann::json& j, const ConfigurationConfirmationMessage& o)
    {
        j = nlohmann::json{{ "seqId", o.seqId() }};
    }

    inline void from_json(const nlohmann::json& j, ConfigurationConfirmationMessage& o)
    {
    }
}

#endif
