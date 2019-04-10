#ifndef COMMUNICATION_MESAGES_INITIALIZATION_CONFIGURATION_POSITION_H
#define COMMUNICATION_MESAGES_INITIALIZATION_CONFIGURATION_POSITION_H

#include <nlohmann/json.hpp>

#include <cstddef>
#include <string>

namespace adaptone
{
    class ConfigurationPosition
    {
    public:
        enum class Type
        {
            Speaker,
            Probe
        };

    private:
        double m_x;
        double m_y;

        Type m_type;

    public:
        ConfigurationPosition();
        ConfigurationPosition(double x, double y, Type type);
        virtual ~ConfigurationPosition();

        double x() const;
        double y() const;

        Type type() const;

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

    inline ConfigurationPosition::Type ConfigurationPosition::type() const
    {
        return m_type;
    }

    inline void to_json(nlohmann::json& j, const ConfigurationPosition& o)
    {
        const char* type = o.m_type == ConfigurationPosition::Type::Speaker ? "s" : "p";
        j = nlohmann::json{{ "x", o.m_x }, { "y", o.m_y }, { "type", type }};
    }

    inline void from_json(const nlohmann::json& j, ConfigurationPosition& o)
    {
        j.at("x").get_to(o.m_x);
        j.at("y").get_to(o.m_y);
        o.m_type = j.at("type") == "s" ? ConfigurationPosition::Type::Speaker : ConfigurationPosition::Type::Probe;
    }
}

#endif
