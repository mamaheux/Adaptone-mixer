#ifndef COMMUNICATION_MESAGES_STOP_PROBE_LISTENING_MESSAGE_H
#define COMMUNICATION_MESAGES_STOP_PROBE_LISTENING_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

namespace adaptone
{
    class StopProbeListeningMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 26;

        StopProbeListeningMessage();
        ~StopProbeListeningMessage() override;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const StopProbeListeningMessage& o);
        friend void from_json(const nlohmann::json& j, StopProbeListeningMessage& o);
    };

    inline std::string StopProbeListeningMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const StopProbeListeningMessage& o)
    {
        j = nlohmann::json{{ "seqId", o.seqId() }};
    }

    inline void from_json(const nlohmann::json& j, StopProbeListeningMessage& o)
    {
    }
}

#endif
