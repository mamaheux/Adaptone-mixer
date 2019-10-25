#ifndef COMMUNICATION_MESAGES_LISTEN_PROBE_MESSAGE_H
#define COMMUNICATION_MESAGES_LISTEN_PROBE_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

#include <cstdint>

namespace adaptone
{
    class ListenProbeMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 25;

    private:
        uint32_t m_probeId;

    public:
        ListenProbeMessage();
        ListenProbeMessage(uint32_t probeId);
        ~ListenProbeMessage() override;

        uint32_t probeId() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const ListenProbeMessage& o);
        friend void from_json(const nlohmann::json& j, ListenProbeMessage& o);
    };

    inline uint32_t ListenProbeMessage::probeId() const
    {
        return m_probeId;
    }

    inline std::string ListenProbeMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const ListenProbeMessage& o)
    {
        nlohmann::json data({{ "probeId", o.m_probeId }});
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, ListenProbeMessage& o)
    {
        j.at("data").at("probeId").get_to(o.m_probeId);
    }
}

#endif
