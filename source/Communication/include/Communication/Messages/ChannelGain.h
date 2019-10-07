#ifndef COMMUNICATION_MESAGES_INPUT_CHANNEL_GAIN_H
#define COMMUNICATION_MESAGES_INPUT_CHANNEL_GAIN_H

#include <nlohmann/json.hpp>

#include <cstddef>

namespace adaptone
{
    class ChannelGain
    {
        std::size_t m_channelId;
        double m_gain;

    public:
        ChannelGain();
        ChannelGain(std::size_t channelId, double gain);
        virtual ~ChannelGain();

        std::size_t channelId() const;
        double gain() const;

        friend void to_json(nlohmann::json& j, const ChannelGain& o);
        friend void from_json(const nlohmann::json& j, ChannelGain& o);
        friend bool operator==(const ChannelGain& l, const ChannelGain& r);
    };

    inline std::size_t ChannelGain::channelId() const
    {
        return m_channelId;
    }

    inline double ChannelGain::gain() const
    {
        return m_gain;
    }

    inline void to_json(nlohmann::json& j, const ChannelGain& o)
    {
        j = nlohmann::json{{ "channelId", o.m_channelId }, { "gain", o.m_gain }};
    }

    inline void from_json(const nlohmann::json& j, ChannelGain& o)
    {
        if (j.contains("data"))
        {
            j.at("data").at("channelId").get_to(o.m_channelId);
            j.at("data").at("gain").get_to(o.m_gain);
        }
        else
        {
            j.at("channelId").get_to(o.m_channelId);
            j.at("gain").get_to(o.m_gain);
        }
    }

    inline bool operator==(const ChannelGain& l, const ChannelGain& r)
    {
        return l.m_channelId == r.m_channelId && l.m_gain == r.m_gain;
    }
}

#endif
