#ifndef COMMUNICATION_MESAGES_INITIALIZATION_CONFIGURATION_CONFIRMATION_MESSAGE_H
#define COMMUNICATION_MESAGES_INITIALIZATION_CONFIGURATION_CONFIRMATION_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

namespace adaptone
{
    class ConfigurationConfirmationMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 9;

        ConfigurationConfirmationMessage();
        ~ConfigurationConfirmationMessage() override;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const ConfigurationConfirmationMessage& o);
        friend void from_json(const nlohmann::json& j, ConfigurationConfirmationMessage& o);
    };

    inline std::string ConfigurationConfirmationMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const ConfigurationConfirmationMessage& o)
    {
        j = nlohmann::json{{ "seqId", o.seqId() }};
    }

    inline void from_json(const nlohmann::json& j, ConfigurationConfirmationMessage& o)
    {
    }
}

#endif
