#ifndef COMMUNICATION_MESAGES_OUTPUT_INPUT_SPECTRUM_MESSAGE_H
#define COMMUNICATION_MESAGES_OUTPUT_INPUT_SPECTRUM_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

#include <Utils/Data/SpectrumPoint.h>

#include <vector>

namespace adaptone
{
    class ChannelSpectrum
    {
        std::size_t m_channelId;
        std::vector<SpectrumPoint> m_points;

    public:
        ChannelSpectrum();
        ChannelSpectrum(std::size_t channelId, const std::vector<SpectrumPoint>& points);
        virtual ~ChannelSpectrum();

        std::size_t channelId() const;
        const std::vector<SpectrumPoint>& points() const;

        friend void to_json(nlohmann::json& j, const ChannelSpectrum& o);
        friend void from_json(const nlohmann::json& j, ChannelSpectrum& o);
    };

    inline std::size_t ChannelSpectrum::channelId() const
    {
        return m_channelId;
    }

    inline const std::vector<SpectrumPoint>& ChannelSpectrum::points() const
    {
        return m_points;
    }

    inline void to_json(nlohmann::json& j, const SpectrumPoint& o)
    {
        j = nlohmann::json{{ "freq", o.frequency() }, { "amplitude", o.amplitude() }};
    }

    inline void from_json(const nlohmann::json& j, SpectrumPoint& o)
    {
        double frequency = 0;
        double amplitude = 0;

        j.at("freq").get_to(frequency);
        j.at("amplitude").get_to(amplitude);

        o = SpectrumPoint(frequency, amplitude);
    }

    inline void to_json(nlohmann::json& j, const ChannelSpectrum& o)
    {
        j = nlohmann::json{{ "channelId", o.m_channelId }, { "points", o.m_points }};
    }

    inline void from_json(const nlohmann::json& j, ChannelSpectrum& o)
    {
        j.at("channelId").get_to(o.m_channelId);
        j.at("points").get_to(o.m_points);
    }

    class InputSpectrumMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 20;

    private:
        std::vector<ChannelSpectrum> m_channelSpectrums;

    public:
        InputSpectrumMessage();
        InputSpectrumMessage(const std::vector<ChannelSpectrum>& channelSpectrums);
        ~InputSpectrumMessage() override;

        const std::vector<ChannelSpectrum>& channelSpectrums() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const InputSpectrumMessage& o);
        friend void from_json(const nlohmann::json& j, InputSpectrumMessage& o);
    };

    inline const std::vector<ChannelSpectrum>& InputSpectrumMessage::channelSpectrums() const
    {
        return m_channelSpectrums;
    }

    inline std::string InputSpectrumMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const InputSpectrumMessage& o)
    {
        nlohmann::json data({{ "spectrums", o.m_channelSpectrums }});
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, InputSpectrumMessage& o)
    {
        j.at("data").at("spectrums").get_to(o.m_channelSpectrums);
    }
}

#endif
