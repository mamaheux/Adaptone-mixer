#ifndef COMMUNICATION_MESAGES_INITIALIZATION_OPTIMIZE_POSITION_MESSAGE_H
#define COMMUNICATION_MESAGES_INITIALIZATION_OPTIMIZE_POSITION_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

namespace adaptone
{
    class OptimizePositionMessage : public ApplicationMessage
    {
    public:
        OptimizePositionMessage();
        virtual ~OptimizePositionMessage();

        friend void to_json(nlohmann::json& j, const OptimizePositionMessage& o);
        friend void from_json(const nlohmann::json& j, OptimizePositionMessage& o);
    };

    inline void to_json(nlohmann::json& j, const OptimizePositionMessage& o)
    {
        j = nlohmann::json{{ "seqId", o.seqId() }};
    }

    inline void from_json(const nlohmann::json& j, OptimizePositionMessage& o)
    {
    }
}

#endif
