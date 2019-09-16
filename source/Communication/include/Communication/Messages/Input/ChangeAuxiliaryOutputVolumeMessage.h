#ifndef COMMUNICATION_MESAGES_INPUT_CHANGE_AUXILIARY_OUTPUT_VOLUME_MESSAGE_H
#define COMMUNICATION_MESAGES_INPUT_CHANGE_AUXILIARY_OUTPUT_VOLUME_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

namespace adaptone
{
    class ChangeAuxiliaryOutputVolumeMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 18;

    private:
        std::size_t m_channelId;
        double m_gain;

    public:
        ChangeAuxiliaryOutputVolumeMessage();
        ChangeAuxiliaryOutputVolumeMessage(std::size_t channelId, double gain);
        ~ChangeAuxiliaryOutputVolumeMessage() override;

        std::size_t channelId() const;
        double gain() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const ChangeAuxiliaryOutputVolumeMessage& o);
        friend void from_json(const nlohmann::json& j, ChangeAuxiliaryOutputVolumeMessage& o);
    };

    inline std::size_t ChangeAuxiliaryOutputVolumeMessage::channelId() const
    {
        return m_channelId;
    }

    inline double ChangeAuxiliaryOutputVolumeMessage::gain() const
    {
        return m_gain;
    }

    inline std::string ChangeAuxiliaryOutputVolumeMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const ChangeAuxiliaryOutputVolumeMessage& o)
    {
        nlohmann::json data({{ "channelId", o.m_channelId },
            { "gain", o.m_gain }
        });
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, ChangeAuxiliaryOutputVolumeMessage& o)
    {
        j.at("data").at("channelId").get_to(o.m_channelId);
        j.at("data").at("gain").get_to(o.m_gain);
    }
}

#endif
