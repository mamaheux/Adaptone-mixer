#ifndef COMMUNICATION_MESAGES_OUTPUT_SOUND_LEVEL_MESSAGE_H
#define COMMUNICATION_MESAGES_OUTPUT_SOUND_LEVEL_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

#include <vector>

namespace adaptone
{
    class ChannelSoundLevel
    {
        std::size_t m_channelId;
        double m_level;
    public:
        ChannelSoundLevel();
        ChannelSoundLevel(std::size_t channelId, double level);
        virtual ~ChannelSoundLevel();

        std::size_t channelId() const;
        double level() const;

        friend void to_json(nlohmann::json& j, const ChannelSoundLevel& o);
        friend void from_json(const nlohmann::json& j, ChannelSoundLevel& o);
        friend bool operator==(const ChannelSoundLevel& l, const ChannelSoundLevel& r);
    };

    inline std::size_t ChannelSoundLevel::channelId() const
    {
        return m_channelId;
    }

    inline double ChannelSoundLevel::level() const
    {
        return m_level;
    }

    inline void to_json(nlohmann::json& j, const ChannelSoundLevel& o)
    {
        j = nlohmann::json{{ "channelId", o.m_channelId }, { "level", o.m_level }};
    }

    inline void from_json(const nlohmann::json& j, ChannelSoundLevel& o)
    {
        j.at("channelId").get_to(o.m_channelId);
        j.at("level").get_to(o.m_level);
    }

    inline bool operator==(const ChannelSoundLevel& l, const ChannelSoundLevel& r)
    {
        return l.m_channelId == r.m_channelId && l.m_level == r.m_level;
    }

    class SoundLevelMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 21;

    private:
        std::vector<ChannelSoundLevel> m_inputAfterGain;
        std::vector<ChannelSoundLevel> m_inputAfterEq;
        std::vector<ChannelSoundLevel> m_outputAfterGain;

    public:
        SoundLevelMessage();
        SoundLevelMessage(const std::vector<ChannelSoundLevel>& inputAfterGain,
            const std::vector<ChannelSoundLevel>& inputAfterEq,
            const std::vector<ChannelSoundLevel>& outputAfterGain);
        ~SoundLevelMessage() override;

        const std::vector<ChannelSoundLevel>& inputAfterGain() const;
        const std::vector<ChannelSoundLevel>& inputAfterEq() const;
        const std::vector<ChannelSoundLevel>& outputAfterGain() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const SoundLevelMessage& o);
        friend void from_json(const nlohmann::json& j, SoundLevelMessage& o);
    };

    inline const std::vector<ChannelSoundLevel>& SoundLevelMessage::inputAfterGain() const
    {
        return m_inputAfterGain;
    }

    inline const std::vector<ChannelSoundLevel>& SoundLevelMessage::inputAfterEq() const
    {
        return m_inputAfterEq;
    }

    inline const std::vector<ChannelSoundLevel>& SoundLevelMessage::outputAfterGain() const
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
