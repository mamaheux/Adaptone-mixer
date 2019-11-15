#ifndef COMMUNICATION_MESAGES_INPUT_TOOGLE_UNIFORMIZATION_MESSAGE_H
#define COMMUNICATION_MESAGES_INPUT_TOOGLE_UNIFORMIZATION_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

#include <cstdint>

namespace adaptone
{
    class ToogleUniformizationMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 27;

    private:
        bool m_isOn;

    public:
        ToogleUniformizationMessage();
        ToogleUniformizationMessage(bool isOn);
        ~ToogleUniformizationMessage() override;

        bool isOn() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const ToogleUniformizationMessage& o);
        friend void from_json(const nlohmann::json& j, ToogleUniformizationMessage& o);
    };

    inline bool ToogleUniformizationMessage::isOn() const
    {
        return m_isOn;
    }

    inline std::string ToogleUniformizationMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const ToogleUniformizationMessage& o)
    {
        nlohmann::json data({{ "isUniformizationOn", o.m_isOn }});
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, ToogleUniformizationMessage& o)
    {
        j.at("data").at("isUniformizationOn").get_to(o.m_isOn);
    }
}

#endif
