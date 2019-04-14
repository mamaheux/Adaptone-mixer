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

        std::size_t m_monitorsNumber;
        std::size_t m_speakersNumber;
        std::size_t m_probesNumber;

        std::vector<ConfigurationPosition> m_positions;

    public:
        ConfigurationChoiceMessage();
        ConfigurationChoiceMessage(std::size_t id,
            const std::string& name,
            std::size_t monitorsNumber,
            std::size_t speakersNumber,
            std::size_t probesNumber,
            const std::vector<ConfigurationPosition>& positions);
        ~ConfigurationChoiceMessage() override;

        std::size_t id() const;
        std::string name() const;

        std::size_t monitorsNumber() const;
        std::size_t speakersNumber() const;
        std::size_t probesNumber() const;

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

    inline std::size_t ConfigurationChoiceMessage::monitorsNumber() const
    {
        return m_monitorsNumber;
    }

    inline std::size_t ConfigurationChoiceMessage::speakersNumber() const
    {
        return m_speakersNumber;
    }

    inline std::size_t ConfigurationChoiceMessage::probesNumber() const
    {
        return m_probesNumber;
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
            { "monitorsNumber", o.m_monitorsNumber },
            { "speakersNumber", o.m_speakersNumber },
            { "probesNumber", o.m_probesNumber },
            { "positions", o.m_positions }
        });
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, ConfigurationChoiceMessage& o)
    {
        j.at("data").at("id").get_to(o.m_id);
        j.at("data").at("name").get_to(o.m_name);
        j.at("data").at("monitorsNumber").get_to(o.m_monitorsNumber);
        j.at("data").at("speakersNumber").get_to(o.m_speakersNumber);
        j.at("data").at("probesNumber").get_to(o.m_probesNumber);
        j.at("data").at("positions").get_to(o.m_positions);
    }
}

#endif
