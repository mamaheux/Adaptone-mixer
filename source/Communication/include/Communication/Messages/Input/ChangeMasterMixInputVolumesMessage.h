#ifndef COMMUNICATION_MESAGES_INPUT_CHANGE_MASTER_MIX_INPUT_VOLUMES_MESSAGE_H
#define COMMUNICATION_MESAGES_INPUT_CHANGE_MASTER_MIX_INPUT_VOLUMES_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>
#include <Communication/Messages/ChannelGain.h>

#include <vector>

namespace adaptone
{
    class ChangeMasterMixInputVolumesMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 14;

    private:
        std::vector<ChannelGain> m_gains;

    public:
        ChangeMasterMixInputVolumesMessage();
        ChangeMasterMixInputVolumesMessage(const std::vector<ChannelGain>& gains);
        ~ChangeMasterMixInputVolumesMessage() override;

        const std::vector<ChannelGain>& gains() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const ChangeMasterMixInputVolumesMessage& o);
        friend void from_json(const nlohmann::json& j, ChangeMasterMixInputVolumesMessage& o);
    };

    inline const std::vector<ChannelGain>& ChangeMasterMixInputVolumesMessage::gains() const
    {
        return m_gains;
    }

    inline std::string ChangeMasterMixInputVolumesMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const ChangeMasterMixInputVolumesMessage& o)
    {
        nlohmann::json data({{ "gains", o.m_gains }});
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, ChangeMasterMixInputVolumesMessage& o)
    {
        j.at("data").at("gains").get_to(o.m_gains);
    }
}

#endif
