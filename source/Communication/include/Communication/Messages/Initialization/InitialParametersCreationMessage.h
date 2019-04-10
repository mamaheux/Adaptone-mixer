#ifndef COMMUNICATION_MESAGES_INITIALIZATION_INITIAL_PARAMETERS_CREATION_MESSAGE_H
#define COMMUNICATION_MESAGES_INITIALIZATION_INITIAL_PARAMETERS_CREATION_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>
#include <Communication/Messages/Initialization/ConfigurationPosition.h>

#include <cstddef>
#include <string>

namespace adaptone
{
    class InitialParametersCreationMessage : public ApplicationMessage
    {
        std::size_t m_id;
        std::string m_name;

        std::size_t m_monitorsNumber;
        std::size_t m_speakersNumber;
        std::size_t m_probesNumber;

    public:
        InitialParametersCreationMessage();
        InitialParametersCreationMessage(std::size_t id,
            const std::string& name,
            std::size_t monitorsNumber,
            std::size_t speakersNumber,
            std::size_t probesNumber);
        virtual ~InitialParametersCreationMessage();

        std::size_t id() const;
        std::string name() const;

        std::size_t monitorsNumber() const;
        std::size_t speakersNumber() const;
        std::size_t probesNumber() const;

        friend void to_json(nlohmann::json& j, const InitialParametersCreationMessage& o);
        friend void from_json(const nlohmann::json& j, InitialParametersCreationMessage& o);
    };

    inline std::size_t InitialParametersCreationMessage::id() const
    {
        return m_id;
    }

    inline std::string InitialParametersCreationMessage::name() const
    {
        return m_name;
    }

    inline std::size_t InitialParametersCreationMessage::monitorsNumber() const
    {
        return m_monitorsNumber;
    }

    inline std::size_t InitialParametersCreationMessage::speakersNumber() const
    {
        return m_speakersNumber;
    }

    inline std::size_t InitialParametersCreationMessage::probesNumber() const
    {
        return m_probesNumber;
    }

    inline void to_json(nlohmann::json& j, const InitialParametersCreationMessage& o)
    {
        nlohmann::json data({{ "id", o.m_id },
            { "name", o.m_name },
            { "monitorsNumber", o.m_monitorsNumber },
            { "speakersNumber", o.m_speakersNumber },
            { "probesNumber", o.m_probesNumber }
        });
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, InitialParametersCreationMessage& o)
    {
        j.at("data").at("id").get_to(o.m_id);
        j.at("data").at("name").get_to(o.m_name);
        j.at("data").at("monitorsNumber").get_to(o.m_monitorsNumber);
        j.at("data").at("speakersNumber").get_to(o.m_speakersNumber);
        j.at("data").at("probesNumber").get_to(o.m_probesNumber);
    }
}

#endif
