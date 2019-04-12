#ifndef COMMUNICATION_MESAGES_INPUT_CHANGE_AUXILIARY_MIX_INPUT_VOLUME_MESSAGE_H
#define COMMUNICATION_MESAGES_INPUT_CHANGE_AUXILIARY_MIX_INPUT_VOLUME_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

#include <Utils/Math.h>

namespace adaptone
{
    class ChangeAuxiliaryMixInputVolumeMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 14;

    private:
        std::size_t m_channelId;
        std::size_t m_auxiliaryId;
        double m_gain;

    public:
        ChangeAuxiliaryMixInputVolumeMessage();
        ChangeAuxiliaryMixInputVolumeMessage(std::size_t channelId, std::size_t auxiliaryId, double gain);
        virtual ~ChangeAuxiliaryMixInputVolumeMessage();

        std::size_t channelId() const;
        std::size_t auxiliaryId() const;
        double gain() const;
        double gainDb() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const ChangeAuxiliaryMixInputVolumeMessage& o);
        friend void from_json(const nlohmann::json& j, ChangeAuxiliaryMixInputVolumeMessage& o);
    };

    inline std::size_t ChangeAuxiliaryMixInputVolumeMessage::channelId() const
    {
        return m_channelId;
    }

    inline std::size_t ChangeAuxiliaryMixInputVolumeMessage::auxiliaryId() const
    {
        return m_auxiliaryId;
    }

    inline double ChangeAuxiliaryMixInputVolumeMessage::gain() const
    {
        return m_gain;
    }

    inline double ChangeAuxiliaryMixInputVolumeMessage::gainDb() const
    {
        return scalarToDb(m_gain);
    }

    inline std::string ChangeAuxiliaryMixInputVolumeMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const ChangeAuxiliaryMixInputVolumeMessage& o)
    {
        nlohmann::json data({{ "channelId", o.m_channelId },
            { "auxiliaryId", o.m_auxiliaryId },
            { "gain", o.m_gain }
        });
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, ChangeAuxiliaryMixInputVolumeMessage& o)
    {
        j.at("data").at("channelId").get_to(o.m_channelId);
        j.at("data").at("auxiliaryId").get_to(o.m_auxiliaryId);
        j.at("data").at("gain").get_to(o.m_gain);
    }
}

#endif
