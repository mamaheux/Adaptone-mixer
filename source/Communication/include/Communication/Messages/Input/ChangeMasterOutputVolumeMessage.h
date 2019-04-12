#ifndef COMMUNICATION_MESAGES_INPUT_CHANGE_MASTER_OUTPUT_VOLUME_MESSAGE_H
#define COMMUNICATION_MESAGES_INPUT_CHANGE_MASTER_OUTPUT_VOLUME_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

#include <Utils/Math.h>

namespace adaptone
{
    class ChangeMasterOutputVolumeMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 17;

    private:
        double m_gain;

    public:
        ChangeMasterOutputVolumeMessage();
        ChangeMasterOutputVolumeMessage(double gain);
        virtual ~ChangeMasterOutputVolumeMessage();

        double gain() const;
        double gainDb() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const ChangeMasterOutputVolumeMessage& o);
        friend void from_json(const nlohmann::json& j, ChangeMasterOutputVolumeMessage& o);
    };

    inline double ChangeMasterOutputVolumeMessage::gain() const
    {
        return m_gain;
    }

    inline double ChangeMasterOutputVolumeMessage::gainDb() const
    {
        return scalarToDb(m_gain);
    }

    inline std::string ChangeMasterOutputVolumeMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const ChangeMasterOutputVolumeMessage& o)
    {
        nlohmann::json data({{ "gain", o.m_gain }});
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, ChangeMasterOutputVolumeMessage& o)
    {
        j.at("data").at("gain").get_to(o.m_gain);
    }
}

#endif
