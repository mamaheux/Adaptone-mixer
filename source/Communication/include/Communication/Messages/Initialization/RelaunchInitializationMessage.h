#ifndef COMMUNICATION_MESAGES_INITIALIZATION_RELAUNCH_INITILIZATION_MESSAGE_H
#define COMMUNICATION_MESAGES_INITIALIZATION_RELAUNCH_INITILIZATION_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

namespace adaptone
{
    class RelaunchInitializationMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 4;

        RelaunchInitializationMessage();
        virtual ~RelaunchInitializationMessage();

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const RelaunchInitializationMessage& o);
        friend void from_json(const nlohmann::json& j, RelaunchInitializationMessage& o);
    };

    inline std::string RelaunchInitializationMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const RelaunchInitializationMessage& o)
    {
        j = nlohmann::json{{ "seqId", o.seqId() }};
    }

    inline void from_json(const nlohmann::json& j, RelaunchInitializationMessage& o)
    {
    }
}

#endif
