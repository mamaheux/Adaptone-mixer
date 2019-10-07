#ifndef COMMUNICATION_MESAGES_INPUT_CHANGE_INPUT_GAINS_MESSAGE_H
#define COMMUNICATION_MESAGES_INPUT_CHANGE_INPUT_GAINS_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>
#include <Communication/Messages/ChannelGain.h>

#include <vector>

namespace adaptone
{
    class ChangeInputGainsMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 11;

    private:
        std::vector<ChannelGain> m_gains;

    public:
        ChangeInputGainsMessage();
        explicit ChangeInputGainsMessage(const std::vector<ChannelGain>& gains);
        ~ChangeInputGainsMessage() override;

        const std::vector<ChannelGain>& gains() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const ChangeInputGainsMessage& o);
        friend void from_json(const nlohmann::json& j, ChangeInputGainsMessage& o);
    };

    inline const std::vector<ChannelGain>& ChangeInputGainsMessage::gains() const
    {
        return m_gains;
    }

    inline std::string ChangeInputGainsMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const ChangeInputGainsMessage& o)
    {
        nlohmann::json data({{ "gains", o.m_gains }});
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, ChangeInputGainsMessage& o)
    {
        j.at("data").at("gains").get_to(o.m_gains);
    }
}

#endif
