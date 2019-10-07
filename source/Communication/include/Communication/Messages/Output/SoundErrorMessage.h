#ifndef COMMUNICATION_MESAGES_OUTPUT_SOUND_ERROR_MESSAGE_H
#define COMMUNICATION_MESAGES_OUTPUT_SOUND_ERROR_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>
#include <Communication/Messages/PositionType.h>

#include <vector>

namespace adaptone
{
    class SoundErrorPosition
    {
        double m_x;
        double m_y;
        PositionType m_type;
        double m_errorRate;

    public:
        SoundErrorPosition();
        SoundErrorPosition(double x, double y, PositionType type, double errorRate);
        virtual ~SoundErrorPosition();

        double x() const;
        double y() const;
        PositionType type() const;
        double errorRate() const;

        friend void to_json(nlohmann::json& j, const SoundErrorPosition& o);
        friend void from_json(const nlohmann::json& j, SoundErrorPosition& o);
    };

    inline double SoundErrorPosition::x() const
    {
        return m_x;
    }

    inline double SoundErrorPosition::y() const
    {
        return m_y;
    }

    inline PositionType SoundErrorPosition::type() const
    {
        return m_type;
    }

    inline double SoundErrorPosition::errorRate() const
    {
        return m_errorRate;
    }

    inline void to_json(nlohmann::json& j, const SoundErrorPosition& o)
    {
        j = nlohmann::json{{ "x", o.m_x }, { "y", o.m_y }, { "type", o.m_type }, { "errorRate", o.m_errorRate }};
    }

    inline void from_json(const nlohmann::json& j, SoundErrorPosition& o)
    {
        j.at("x").get_to(o.m_x);
        j.at("y").get_to(o.m_y);
        j.at("type").get_to(o.m_type);
        j.at("errorRate").get_to(o.m_errorRate);
    }

    class SoundErrorMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 21;

    private:
        std::vector<SoundErrorPosition> m_positions;

    public:
        SoundErrorMessage();
        explicit SoundErrorMessage(const std::vector<SoundErrorPosition>& positions);
        ~SoundErrorMessage() override;

        const std::vector<SoundErrorPosition>& positions() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const SoundErrorMessage& o);
        friend void from_json(const nlohmann::json& j, SoundErrorMessage& o);
    };

    inline const std::vector<SoundErrorPosition>& SoundErrorMessage::positions() const
    {
        return m_positions;
    }

    inline std::string SoundErrorMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const SoundErrorMessage& o)
    {
        nlohmann::json data({{ "positions", o.m_positions }});
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, SoundErrorMessage& o)
    {
        j.at("data").at("positions").get_to(o.m_positions);
    }
}

#endif
