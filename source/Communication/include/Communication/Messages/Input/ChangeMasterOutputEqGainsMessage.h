#ifndef COMMUNICATION_MESAGES_INPUT_CHANGE_MASTER_OUTPUT_EQ_GAINS_MESSAGE_H
#define COMMUNICATION_MESAGES_INPUT_CHANGE_MASTER_OUTPUT_EQ_GAINS_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

#include <vector>

namespace adaptone
{
    class ChangeMasterOutputEqGainsMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 17;

    private:
        std::vector<double> m_gains;

    public:
        ChangeMasterOutputEqGainsMessage();
        ChangeMasterOutputEqGainsMessage(const std::vector<double>& gains);
        ~ChangeMasterOutputEqGainsMessage() override;

        const std::vector<double>& gains() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const ChangeMasterOutputEqGainsMessage& o);
        friend void from_json(const nlohmann::json& j, ChangeMasterOutputEqGainsMessage& o);
    };

    inline const std::vector<double>& ChangeMasterOutputEqGainsMessage::gains() const
    {
        return m_gains;
    }

    inline std::string ChangeMasterOutputEqGainsMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const ChangeMasterOutputEqGainsMessage& o)
    {
        nlohmann::json data({{ "gains", o.m_gains }});
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, ChangeMasterOutputEqGainsMessage& o)
    {
        j.at("data").at("gains").get_to(o.m_gains);
    }
}

#endif
