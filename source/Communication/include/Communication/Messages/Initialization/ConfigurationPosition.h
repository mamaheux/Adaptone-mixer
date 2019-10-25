#ifndef COMMUNICATION_MESAGES_INITIALIZATION_CONFIGURATION_POSITION_H
#define COMMUNICATION_MESAGES_INITIALIZATION_CONFIGURATION_POSITION_H

#include <Communication/Messages/PositionType.h>

#include <nlohmann/json.hpp>

#include <cstddef>

namespace adaptone
{
    class ConfigurationPosition
    {
        double m_x;
        double m_y;

        PositionType m_type;
        uint32_t m_id;

    public:
        ConfigurationPosition();
        ConfigurationPosition(double x, double y, PositionType type, uint32_t id);
        virtual ~ConfigurationPosition();

        double x() const;
        double y() const;

        PositionType type() const;
        uint32_t id() const;

        friend void to_json(nlohmann::json& j, const ConfigurationPosition& o);
        friend void from_json(const nlohmann::json& j, ConfigurationPosition& o);
    };

    inline double ConfigurationPosition::x() const
    {
        return m_x;
    }

    inline double ConfigurationPosition::y() const
    {
        return m_y;
    }

    inline PositionType ConfigurationPosition::type() const
    {
        return m_type;
    }

    inline uint32_t ConfigurationPosition::id() const
    {
        return m_id;
    }

    inline void to_json(nlohmann::json& j, const ConfigurationPosition& o)
    {
        j = nlohmann::json{{ "x", o.m_x }, { "y", o.m_y }, { "type", o.m_type }, { "id", o.m_id }};
    }

    inline void from_json(const nlohmann::json& j, ConfigurationPosition& o)
    {
        j.at("x").get_to(o.m_x);
        j.at("y").get_to(o.m_y);
        j.at("type").get_to(o.m_type);
        j.at("id").get_to(o.m_id);
    }
}

#endif
