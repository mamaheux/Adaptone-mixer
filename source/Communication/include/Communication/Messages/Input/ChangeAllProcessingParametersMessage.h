#ifndef COMMUNICATION_MESAGES_INPUT_CHANGE_ALL_PROCESSING_PARAMETERS_MESSAGE_H
#define COMMUNICATION_MESAGES_INPUT_CHANGE_ALL_PROCESSING_PARAMETERS_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>
#include <Communication/Messages/ChannelGain.h>

#include <vector>

namespace adaptone
{
    class InputProcessingParameters
    {
        std::size_t m_channelId;
        double m_gain;
        bool m_isMuted;
        bool m_isSolo;
        std::vector<double> m_eqGains;

    public:
        InputProcessingParameters();
        InputProcessingParameters(size_t channelId,
            double gain,
            bool isMuted,
            bool isSolo,
            std::vector<double> eqGains);
        virtual ~InputProcessingParameters();

        std::size_t channelId() const;
        double gain() const;
        bool isMuted() const;
        bool isSolo() const;
        const std::vector<double>& eqGains() const;

        friend void to_json(nlohmann::json& j, const InputProcessingParameters& o);
        friend void from_json(const nlohmann::json& j, InputProcessingParameters& o);
        friend bool operator==(const InputProcessingParameters& l, const InputProcessingParameters& r);
    };

    inline std::size_t InputProcessingParameters::channelId() const
    {
        return m_channelId;
    }

    inline double InputProcessingParameters::gain() const
    {
        return m_gain;
    }

    inline bool InputProcessingParameters::isMuted() const
    {
        return m_isMuted;
    }

    inline bool InputProcessingParameters::isSolo() const
    {
        return m_isSolo;
    }

    inline const std::vector<double>& InputProcessingParameters::eqGains() const
    {
        return m_eqGains;
    }

    inline void to_json(nlohmann::json& j, const InputProcessingParameters& o)
    {
        nlohmann::json data{{ "channelId", o.m_channelId },
            { "gain", o.m_gain },
            { "isMuted", o.m_isMuted },
            { "isSolo", o.m_isSolo },
            { "eqGains", o.m_eqGains }};

        j = nlohmann::json{{ "data", data }};
    }

    inline void from_json(const nlohmann::json& j, InputProcessingParameters& o)
    {
        j.at("data").at("channelId").get_to(o.m_channelId);
        j.at("data").at("gain").get_to(o.m_gain);
        j.at("data").at("isMuted").get_to(o.m_isMuted);
        j.at("data").at("isSolo").get_to(o.m_isSolo);
        j.at("data").at("eqGains").get_to(o.m_eqGains);
    }

    inline bool operator==(const InputProcessingParameters& l, const InputProcessingParameters& r)
    {
        return l.m_channelId == r.m_channelId &&
            l.m_gain == r.m_gain &&
            l.m_isMuted ==  r.m_isMuted &&
            l.m_isSolo == r.m_isSolo &&
            l.m_eqGains == r.m_eqGains;
    }

    class MasterProcessingParameters
    {
        double m_gain;
        bool m_isMuted;
        std::vector<ChannelGain> m_inputs;
        std::vector<double> m_eqGains;

    public:
        MasterProcessingParameters();
        MasterProcessingParameters(double gain,
            bool isMuted,
            std::vector<ChannelGain> inputs,
            std::vector<double> eqGains);
        virtual ~MasterProcessingParameters();

        double gain() const;
        bool isMuted() const;
        const std::vector<ChannelGain>& inputs() const;
        const std::vector<double>& eqGains() const;

        friend void to_json(nlohmann::json& j, const MasterProcessingParameters& o);
        friend void from_json(const nlohmann::json& j, MasterProcessingParameters& o);
        friend bool operator==(const MasterProcessingParameters& l, const MasterProcessingParameters& r);
    };

    inline double MasterProcessingParameters::gain() const
    {
        return m_gain;
    }

    inline bool MasterProcessingParameters::isMuted() const
    {
        return m_isMuted;
    }

    inline const std::vector<ChannelGain>& MasterProcessingParameters::inputs() const
    {
        return m_inputs;
    }

    inline const std::vector<double>& MasterProcessingParameters::eqGains() const
    {
        return m_eqGains;
    }

    inline void to_json(nlohmann::json& j, const MasterProcessingParameters& o)
    {
        nlohmann::json data{{ "gain", o.m_gain },
            { "isMuted", o.m_isMuted },
            { "inputs", o.m_inputs },
            { "eqGains", o.m_eqGains }};

        j = nlohmann::json{{ "data", data }};
    }

    inline void from_json(const nlohmann::json& j, MasterProcessingParameters& o)
    {
        j.at("data").at("gain").get_to(o.m_gain);
        j.at("data").at("isMuted").get_to(o.m_isMuted);
        j.at("data").at("inputs").get_to(o.m_inputs);
        j.at("data").at("eqGains").get_to(o.m_eqGains);
    }

    inline bool operator==(const MasterProcessingParameters& l, const MasterProcessingParameters& r)
    {
        return l.m_gain == r.m_gain &&
            l.m_isMuted ==  r.m_isMuted &&
            l.m_inputs == r.m_inputs &&
            l.m_eqGains == r.m_eqGains;
    }

    class AuxiliaryProcessingParameters
    {
        std::size_t m_auxiliaryChannelId;
        double m_gain;
        bool m_isMuted;
        std::vector<ChannelGain> m_inputs;
        std::vector<double> m_eqGains;

    public:
        AuxiliaryProcessingParameters();
        AuxiliaryProcessingParameters(std::size_t auxiliaryChannelId,
            double gain,
            bool isMuted,
            std::vector<ChannelGain> inputs,
            std::vector<double> eqGains);
        virtual ~AuxiliaryProcessingParameters();

        std::size_t auxiliaryChannelId() const;
        double gain() const;
        bool isMuted() const;
        const std::vector<ChannelGain>& inputs() const;
        const std::vector<double>& eqGains() const;

        friend void to_json(nlohmann::json& j, const AuxiliaryProcessingParameters& o);
        friend void from_json(const nlohmann::json& j, AuxiliaryProcessingParameters& o);
        friend bool operator==(const AuxiliaryProcessingParameters& l, const AuxiliaryProcessingParameters& r);
    };

    inline std::size_t AuxiliaryProcessingParameters::auxiliaryChannelId() const
    {
        return m_auxiliaryChannelId;
    }

    inline double AuxiliaryProcessingParameters::gain() const
    {
        return m_gain;
    }

    inline bool AuxiliaryProcessingParameters::isMuted() const
    {
        return m_isMuted;
    }

    inline const std::vector<ChannelGain>& AuxiliaryProcessingParameters::inputs() const
    {
        return m_inputs;
    }

    inline const std::vector<double>& AuxiliaryProcessingParameters::eqGains() const
    {
        return m_eqGains;
    }

    inline void to_json(nlohmann::json& j, const AuxiliaryProcessingParameters& o)
    {
        nlohmann::json data{{ "channelId", o.m_auxiliaryChannelId },
            { "gain", o.m_gain },
            { "isMuted", o.m_isMuted },
            { "inputs", o.m_inputs },
            { "eqGains", o.m_eqGains }};

        j = nlohmann::json{{ "data", data }};
    }

    inline void from_json(const nlohmann::json& j, AuxiliaryProcessingParameters& o)
    {
        j.at("data").at("channelId").get_to(o.m_auxiliaryChannelId);
        j.at("data").at("gain").get_to(o.m_gain);
        j.at("data").at("isMuted").get_to(o.m_isMuted);
        j.at("data").at("inputs").get_to(o.m_inputs);
        j.at("data").at("eqGains").get_to(o.m_eqGains);
    }

    inline bool operator==(const AuxiliaryProcessingParameters& l, const AuxiliaryProcessingParameters& r)
    {
        return l.m_auxiliaryChannelId == r.m_auxiliaryChannelId &&
            l.m_gain == r.m_gain &&
            l.m_isMuted ==  r.m_isMuted &&
            l.m_inputs == r.m_inputs &&
            l.m_eqGains == r.m_eqGains;
    }

    class ChangeAllProcessingParametersMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 24;

    private:
        std::vector<InputProcessingParameters> m_inputs;
        MasterProcessingParameters m_master;
        std::vector<AuxiliaryProcessingParameters> m_auxiliaries;
        std::vector<std::size_t> m_inputChannelIds;
        std::size_t m_speakersNumber;
        std::vector<std::size_t> m_auxiliaryChannelIds;

    public:
        ChangeAllProcessingParametersMessage();
        ChangeAllProcessingParametersMessage(std::vector<InputProcessingParameters> inputs,
            MasterProcessingParameters master,
            std::vector<AuxiliaryProcessingParameters> auxiliaries,
            std::vector<std::size_t> inputChannelIds,
            std::size_t speakersNumber,
            std::vector<std::size_t> auxiliaryChannelIds);
        ~ChangeAllProcessingParametersMessage() override;

        const std::vector<InputProcessingParameters>& inputs() const;
        const MasterProcessingParameters& master() const;
        const std::vector<AuxiliaryProcessingParameters>& auxiliaries() const;
        const std::vector<std::size_t>& inputChannelIds() const;
        const std::size_t speakersNumber() const;
        const std::vector<std::size_t>& auxiliaryChannelIds() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const ChangeAllProcessingParametersMessage& o);
        friend void from_json(const nlohmann::json& j, ChangeAllProcessingParametersMessage& o);
    };

    inline const std::vector<InputProcessingParameters>& ChangeAllProcessingParametersMessage::inputs() const
    {
        return m_inputs;
    }

    inline const MasterProcessingParameters& ChangeAllProcessingParametersMessage::master() const
    {
        return m_master;
    }

    inline const std::vector<AuxiliaryProcessingParameters>& ChangeAllProcessingParametersMessage::auxiliaries() const
    {
        return m_auxiliaries;
    }

    inline const std::vector<std::size_t>& ChangeAllProcessingParametersMessage::inputChannelIds() const
    {
        return m_inputChannelIds;
    }

    inline const std::size_t ChangeAllProcessingParametersMessage::speakersNumber() const
    {
        return m_speakersNumber;
    }

    inline const std::vector<std::size_t>& ChangeAllProcessingParametersMessage::auxiliaryChannelIds() const
    {
        return m_auxiliaryChannelIds;
    }

    inline std::string ChangeAllProcessingParametersMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const ChangeAllProcessingParametersMessage& o)
    {
        nlohmann::json channels({{ "inputs", o.m_inputs },
            { "master", o.m_master },
            { "auxiliaries", o.m_auxiliaries },
            { "inputChannelIds", o.m_inputChannelIds },
            { "speakersNumber", o.m_speakersNumber },
            { "auxiliaryChannelIds", o.m_auxiliaryChannelIds }
        });
        nlohmann::json data({{ "channels", channels }});
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, ChangeAllProcessingParametersMessage& o)
    {
        j.at("data").at("channels").at("inputs").get_to(o.m_inputs);
        j.at("data").at("channels").at("master").get_to(o.m_master);
        j.at("data").at("channels").at("auxiliaries").get_to(o.m_auxiliaries);
        j.at("data").at("inputChannelIds").get_to(o.m_inputChannelIds);
        j.at("data").at("speakersNumber").get_to(o.m_speakersNumber);
        j.at("data").at("auxiliaryChannelIds").get_to(o.m_auxiliaryChannelIds);
    }
}

#endif
