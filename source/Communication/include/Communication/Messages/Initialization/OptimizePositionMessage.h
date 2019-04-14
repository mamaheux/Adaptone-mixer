#ifndef COMMUNICATION_MESAGES_INITIALIZATION_OPTIMIZE_POSITION_MESSAGE_H
#define COMMUNICATION_MESAGES_INITIALIZATION_OPTIMIZE_POSITION_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

namespace adaptone
{
    class OptimizePositionMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 6;

        OptimizePositionMessage();
        ~OptimizePositionMessage() override;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const OptimizePositionMessage& o);
        friend void from_json(const nlohmann::json& j, OptimizePositionMessage& o);
    };

    inline std::string OptimizePositionMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const OptimizePositionMessage& o)
    {
        j = nlohmann::json{{ "seqId", o.seqId() }};
    }

    inline void from_json(const nlohmann::json& j, OptimizePositionMessage& o)
    {
    }
}

#endif
