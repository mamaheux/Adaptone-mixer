#ifndef COMMUNICATION_MESAGES_INITIALIZATION_CONFIGURATION_CHOICE_MESSAGE_H
#define COMMUNICATION_MESAGES_INITIALIZATION_CONFIGURATION_CHOICE_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>
#include <Communication/Messages/Initialization/ConfigurationPosition.h>

#include <cstddef>
#include <string>

namespace adaptone
{
    class ConfigurationChoiceMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 0;

    private:
        std::size_t m_id;
        std::string m_name;

        std::vector<std::size_t> m_inputChannelIds;
        std::size_t m_speakersNumber;
        std::vector<std::size_t> m_auxiliaryChannelIds;

        std::vector<ConfigurationPosition> m_positions;

    public:
        ConfigurationChoiceMessage();
        ConfigurationChoiceMessage(std::size_t id,
            const std::string& name,
            const std::vector<std::size_t>& inputChannelIds,
            std::size_t speakersNumber,
            const std::vector<std::size_t>& auxiliaryChannelIds,
            const std::vector<ConfigurationPosition>& positions);
        ~ConfigurationChoiceMessage() override;

        std::size_t id() const;
        std::string name() const;

        const std::vector<std::size_t>& inputChannelIds() const;
        std::size_t speakersNumber() const;
        const std::vector<std::size_t>& auxiliaryChannelIds() const;

        const std::vector<ConfigurationPosition>& positions() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const ConfigurationChoiceMessage& o);
        friend void from_json(const nlohmann::json& j, ConfigurationChoiceMessage& o);
    };

    inline std::size_t ConfigurationChoiceMessage::id() const
    {
        return m_id;
    }

    inline std::string ConfigurationChoiceMessage::name() const
    {
        return m_name;
    }

    inline const std::vector<std::size_t>& ConfigurationChoiceMessage::inputChannelIds() const
    {
        return m_inputChannelIds;
    }

    inline std::size_t ConfigurationChoiceMessage::speakersNumber() const
    {
        return m_speakersNumber;
    }

    inline const std::vector<std::size_t>& ConfigurationChoiceMessage::auxiliaryChannelIds() const
    {
        return m_auxiliaryChannelIds;
    }

    inline const std::vector<ConfigurationPosition>& ConfigurationChoiceMessage::positions() const
    {
        return m_positions;
    }

    inline std::string ConfigurationChoiceMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const ConfigurationChoiceMessage& o)
    {
        nlohmann::json data({{ "id", o.m_id },
            { "name", o.m_name },
            { "inputChannelIds", o.m_inputChannelIds },
            { "speakersNumber", o.m_speakersNumber },
            { "auxiliaryChannelIds", o.m_auxiliaryChannelIds },
            { "positions", o.m_positions }
        });
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, ConfigurationChoiceMessage& o)
    {
        j.at("data").at("id").get_to(o.m_id);
        j.at("data").at("name").get_to(o.m_name);
        j.at("data").at("inputChannelIds").get_to(o.m_inputChannelIds);
        j.at("data").at("speakersNumber").get_to(o.m_speakersNumber);
        j.at("data").at("auxiliaryChannelIds").get_to(o.m_auxiliaryChannelIds);
        j.at("data").at("positions").get_to(o.m_positions);
    }
}

#endif
