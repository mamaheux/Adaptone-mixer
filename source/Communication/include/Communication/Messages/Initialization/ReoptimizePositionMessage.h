#ifndef COMMUNICATION_MESAGES_INITIALIZATION_OPTIMIZE_POSITION_MESSAGE_H
#define COMMUNICATION_MESAGES_INITIALIZATION_OPTIMIZE_POSITION_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

namespace adaptone
{
    class ReoptimizePositionMessage : public ApplicationMessage
    {
    public:
        ReoptimizePositionMessage();
        virtual ~ReoptimizePositionMessage();

        friend void to_json(nlohmann::json& j, const ReoptimizePositionMessage& o);
        friend void from_json(const nlohmann::json& j, ReoptimizePositionMessage& o);
    };

    inline void to_json(nlohmann::json& j, const ReoptimizePositionMessage& o)
    {
        j = nlohmann::json{{ "seqId", o.seqId() }};
    }

    inline void from_json(const nlohmann::json& j, ReoptimizePositionMessage& o)
    {
    }
}

#endif
