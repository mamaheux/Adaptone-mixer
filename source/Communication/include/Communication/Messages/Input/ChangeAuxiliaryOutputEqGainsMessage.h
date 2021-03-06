#ifndef COMMUNICATION_MESAGES_INPUT_CHANGE_AUXILIARY_OUTPUT_EQ_GAINS_MESSAGE_H
#define COMMUNICATION_MESAGES_INPUT_CHANGE_AUXILIARY_OUTPUT_EQ_GAINS_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

namespace adaptone
{
    class ChangeAuxiliaryOutputEqGainsMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 18;

    private:
        std::size_t m_channelId;
        std::vector<double> m_gains;

    public:
        ChangeAuxiliaryOutputEqGainsMessage();
        ChangeAuxiliaryOutputEqGainsMessage(std::size_t channelId, std::vector<double> gains);
        ~ChangeAuxiliaryOutputEqGainsMessage() override;

        std::size_t channelId() const;
        const std::vector<double>& gains() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const ChangeAuxiliaryOutputEqGainsMessage& o);
        friend void from_json(const nlohmann::json& j, ChangeAuxiliaryOutputEqGainsMessage& o);
    };

    inline std::size_t ChangeAuxiliaryOutputEqGainsMessage::channelId() const
    {
        return m_channelId;
    }

    inline const std::vector<double>& ChangeAuxiliaryOutputEqGainsMessage::gains() const
    {
        return m_gains;
    }

    inline std::string ChangeAuxiliaryOutputEqGainsMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const ChangeAuxiliaryOutputEqGainsMessage& o)
    {
        nlohmann::json data({{ "channelId", o.m_channelId },
            { "gains", o.m_gains }
        });
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, ChangeAuxiliaryOutputEqGainsMessage& o)
    {
        j.at("data").at("channelId").get_to(o.m_channelId);
        j.at("data").at("gains").get_to(o.m_gains);
    }
}

#endif
