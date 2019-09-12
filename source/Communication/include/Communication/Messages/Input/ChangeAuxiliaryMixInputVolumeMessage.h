#ifndef COMMUNICATION_MESAGES_INPUT_CHANGE_AUXILIARY_MIX_INPUT_VOLUME_MESSAGE_H
#define COMMUNICATION_MESAGES_INPUT_CHANGE_AUXILIARY_MIX_INPUT_VOLUME_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

namespace adaptone
{
    class ChangeAuxiliaryMixInputVolumeMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 14;

    private:
        std::size_t m_channelId;
        std::size_t m_auxiliaryChannelId;
        double m_gain;

    public:
        ChangeAuxiliaryMixInputVolumeMessage();
        ChangeAuxiliaryMixInputVolumeMessage(std::size_t channelId, std::size_t auxiliaryChannelId, double gain);
        ~ChangeAuxiliaryMixInputVolumeMessage() override;

        std::size_t channelId() const;
        std::size_t auxiliaryChannelId() const;
        double gain() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const ChangeAuxiliaryMixInputVolumeMessage& o);
        friend void from_json(const nlohmann::json& j, ChangeAuxiliaryMixInputVolumeMessage& o);
    };

    inline std::size_t ChangeAuxiliaryMixInputVolumeMessage::channelId() const
    {
        return m_channelId;
    }

    inline std::size_t ChangeAuxiliaryMixInputVolumeMessage::auxiliaryChannelId() const
    {
        return m_auxiliaryChannelId;
    }

    inline double ChangeAuxiliaryMixInputVolumeMessage::gain() const
    {
        return m_gain;
    }

    inline std::string ChangeAuxiliaryMixInputVolumeMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const ChangeAuxiliaryMixInputVolumeMessage& o)
    {
        nlohmann::json data({{ "channelId", o.m_channelId },
            { "auxiliaryChannelId", o.m_auxiliaryChannelId },
            { "gain", o.m_gain }
        });
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, ChangeAuxiliaryMixInputVolumeMessage& o)
    {
        j.at("data").at("channelId").get_to(o.m_channelId);
        j.at("data").at("auxiliaryChannelId").get_to(o.m_auxiliaryChannelId);
        j.at("data").at("gain").get_to(o.m_gain);
    }
}

#endif
