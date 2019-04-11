#ifndef COMMUNICATION_MESAGES_INPUT_CHANGE_INPUT_EQ_GAINS_MESSAGE_H
#define COMMUNICATION_MESAGES_INPUT_CHANGE_INPUT_EQ_GAINS_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

#include <Utils/Math.h>

#include <vector>

namespace adaptone
{
    class ChangeInputEqGainsMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 12;

    private:
        std::size_t m_channelId;
        std::vector<double> m_gains;

    public:
        ChangeInputEqGainsMessage();
        ChangeInputEqGainsMessage(std::size_t channelId, const std::vector<double>& gains);
        virtual ~ChangeInputEqGainsMessage();

        std::size_t channelId() const;
        const std::vector<double>& gains() const;
        std::vector<double> gainsDb() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const ChangeInputEqGainsMessage& o);
        friend void from_json(const nlohmann::json& j, ChangeInputEqGainsMessage& o);
    };

    inline std::size_t ChangeInputEqGainsMessage::channelId() const
    {
        return m_channelId;
    }

    inline const std::vector<double>& ChangeInputEqGainsMessage::gains() const
    {
        return m_gains;
    }

    inline std::vector<double> ChangeInputEqGainsMessage::gainsDb() const
    {
        return vectorToDb(m_gains);
    }

    inline std::string ChangeInputEqGainsMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const ChangeInputEqGainsMessage& o)
    {
        nlohmann::json data({{ "channelId", o.m_channelId },
            { "gains", o.m_gains }
        });
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, ChangeInputEqGainsMessage& o)
    {
        j.at("data").at("channelId").get_to(o.m_channelId);
        j.at("data").at("gains").get_to(o.m_gains);
    }
}

#endif
