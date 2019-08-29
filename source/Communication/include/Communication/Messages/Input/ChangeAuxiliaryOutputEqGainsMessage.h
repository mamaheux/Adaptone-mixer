#ifndef COMMUNICATION_MESAGES_INPUT_CHANGE_AUXILIARY_OUTPUT_EQ_GAINS_MESSAGE_H
#define COMMUNICATION_MESAGES_INPUT_CHANGE_AUXILIARY_OUTPUT_EQ_GAINS_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

namespace adaptone
{
    class ChangeAuxiliaryOutputEqGainsMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 16;

    private:
        std::size_t m_auxiliaryId;
        std::vector<double> m_gains;

    public:
        ChangeAuxiliaryOutputEqGainsMessage();
        ChangeAuxiliaryOutputEqGainsMessage(std::size_t auxiliaryId, const std::vector<double>& gains);
        ~ChangeAuxiliaryOutputEqGainsMessage() override;

        std::size_t auxiliaryId() const;
        const std::vector<double>& gains() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const ChangeAuxiliaryOutputEqGainsMessage& o);
        friend void from_json(const nlohmann::json& j, ChangeAuxiliaryOutputEqGainsMessage& o);
    };

    inline std::size_t ChangeAuxiliaryOutputEqGainsMessage::auxiliaryId() const
    {
        return m_auxiliaryId;
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
        nlohmann::json data({{ "auxiliaryId", o.m_auxiliaryId },
            { "gains", o.m_gains }
        });
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, ChangeAuxiliaryOutputEqGainsMessage& o)
    {
        j.at("data").at("auxiliaryId").get_to(o.m_auxiliaryId);
        j.at("data").at("gains").get_to(o.m_gains);
    }
}

#endif
