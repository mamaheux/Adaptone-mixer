#ifndef COMMUNICATION_MESAGES_INPUT_CHANGE_MASTER_MIX_INPUT_VOLUME_MESSAGE_H
#define COMMUNICATION_MESAGES_INPUT_CHANGE_MASTER_MIX_INPUT_VOLUME_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

#include <Utils/Math.h>

namespace adaptone
{
    class ChangeMasterMixInputVolumeMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 13;

    private:
        std::size_t m_channelId;
        double m_gain;

    public:
        ChangeMasterMixInputVolumeMessage();
        ChangeMasterMixInputVolumeMessage(std::size_t channelId, double gain);
        ~ChangeMasterMixInputVolumeMessage() override;

        std::size_t channelId() const;
        double gain() const;
        double gainDb() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const ChangeMasterMixInputVolumeMessage& o);
        friend void from_json(const nlohmann::json& j, ChangeMasterMixInputVolumeMessage& o);
    };

    inline std::size_t ChangeMasterMixInputVolumeMessage::channelId() const
    {
        return m_channelId;
    }

    inline double ChangeMasterMixInputVolumeMessage::gain() const
    {
        return m_gain;
    }

    inline double ChangeMasterMixInputVolumeMessage::gainDb() const
    {
        return scalarToDb(m_gain);
    }

    inline std::string ChangeMasterMixInputVolumeMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const ChangeMasterMixInputVolumeMessage& o)
    {
        nlohmann::json data({{ "channelId", o.m_channelId },
            { "gain", o.m_gain }
        });
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, ChangeMasterMixInputVolumeMessage& o)
    {
        j.at("data").at("channelId").get_to(o.m_channelId);
        j.at("data").at("gain").get_to(o.m_gain);
    }
}

#endif
