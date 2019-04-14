#ifndef COMMUNICATION_MESAGES_OUTPUT_SOUND_LEVEL_MESSAGE_H
#define COMMUNICATION_MESAGES_OUTPUT_SOUND_LEVEL_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

#include <vector>

namespace adaptone
{
    class SoundLevelMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 21;

    private:
        std::vector<double> m_inputAfterGain;
        std::vector<double> m_inputAfterEq;
        std::vector<double> m_outputAfterGain;

    public:
        SoundLevelMessage();
        SoundLevelMessage(const std::vector<double>& inputAfterGain,
            const std::vector<double>& inputAfterEq,
            const std::vector<double>& outputAfterGain);
        ~SoundLevelMessage() override;

        const std::vector<double>& inputAfterGain() const;
        const std::vector<double>& inputAfterEq() const;
        const std::vector<double>& outputAfterGain() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const SoundLevelMessage& o);
        friend void from_json(const nlohmann::json& j, SoundLevelMessage& o);
    };

    inline const std::vector<double>& SoundLevelMessage::inputAfterGain() const
    {
        return m_inputAfterGain;
    }

    inline const std::vector<double>& SoundLevelMessage::inputAfterEq() const
    {
        return m_inputAfterEq;
    }

    inline const std::vector<double>& SoundLevelMessage::outputAfterGain() const
    {
        return m_outputAfterGain;
    }

    inline std::string SoundLevelMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const SoundLevelMessage& o)
    {
        nlohmann::json data({{ "inputAfterGain", o.m_inputAfterGain },
            { "inputAfterEq", o.m_inputAfterEq },
            { "outputAfterGain", o.m_outputAfterGain }
        });
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, SoundLevelMessage& o)
    {
        j.at("data").at("inputAfterGain").get_to(o.m_inputAfterGain);
        j.at("data").at("inputAfterEq").get_to(o.m_inputAfterEq);
        j.at("data").at("outputAfterGain").get_to(o.m_outputAfterGain);
    }
}

#endif
