#ifndef COMMUNICATION_MESAGES_INITIALIZATION_REOPTIMIZE_POSITION_MESSAGE_H
#define COMMUNICATION_MESAGES_INITIALIZATION_REOPTIMIZE_POSITION_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

namespace adaptone
{
    class ReoptimizePositionMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 8;

        ReoptimizePositionMessage();
        ~ReoptimizePositionMessage() override;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const ReoptimizePositionMessage& o);
        friend void from_json(const nlohmann::json& j, ReoptimizePositionMessage& o);
    };

    inline std::string ReoptimizePositionMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const ReoptimizePositionMessage& o)
    {
        j = nlohmann::json{{ "seqId", o.seqId() }};
    }

    inline void from_json(const nlohmann::json& j, ReoptimizePositionMessage& o)
    {
    }
}

#endif
