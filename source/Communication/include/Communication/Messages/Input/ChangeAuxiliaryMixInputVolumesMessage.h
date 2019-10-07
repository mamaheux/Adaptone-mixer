#ifndef COMMUNICATION_MESAGES_INPUT_CHANGE_AUXILIARY_MIX_INPUT_VOLUMES_MESSAGE_H
#define COMMUNICATION_MESAGES_INPUT_CHANGE_AUXILIARY_MIX_INPUT_VOLUMES_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>
#include <Communication/Messages/ChannelGain.h>

#include <vector>

namespace adaptone
{
    class ChangeAuxiliaryMixInputVolumesMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 16;

    private:
        std::size_t m_auxiliaryChannelId;
        std::vector<ChannelGain> m_gains;

    public:
        ChangeAuxiliaryMixInputVolumesMessage();
        ChangeAuxiliaryMixInputVolumesMessage(std::size_t auxiliaryChannelId, const std::vector<ChannelGain>& gains);
        ~ChangeAuxiliaryMixInputVolumesMessage() override;

        std::size_t auxiliaryChannelId() const;
        const std::vector<ChannelGain>& gains() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const ChangeAuxiliaryMixInputVolumesMessage& o);
        friend void from_json(const nlohmann::json& j, ChangeAuxiliaryMixInputVolumesMessage& o);
    };

    inline std::size_t ChangeAuxiliaryMixInputVolumesMessage::auxiliaryChannelId() const
    {
        return m_auxiliaryChannelId;
    }

    inline const std::vector<ChannelGain>& ChangeAuxiliaryMixInputVolumesMessage::gains() const
    {
        return m_gains;
    }

    inline std::string ChangeAuxiliaryMixInputVolumesMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const ChangeAuxiliaryMixInputVolumesMessage& o)
    {
        nlohmann::json data({{ "auxiliaryChannelId", o.m_auxiliaryChannelId },
            { "gains", o.m_gains }
        });
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, ChangeAuxiliaryMixInputVolumesMessage& o)
    {
        j.at("data").at("auxiliaryChannelId").get_to(o.m_auxiliaryChannelId);
        j.at("data").at("gains").get_to(o.m_gains);
    }
}

#endif
