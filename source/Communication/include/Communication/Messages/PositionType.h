#ifndef COMMUNICATION_MESAGES_INPUT_POSITION_TYPE_H
#define COMMUNICATION_MESAGES_INPUT_POSITION_TYPE_H

#include <Utils/Exception/NotSupportedException.h>

#include <nlohmann/json.hpp>

namespace adaptone
{
    enum class PositionType
    {
        Speaker,
        Probe
    };

    inline void to_json(nlohmann::json& j, const PositionType& o)
    {
        switch (o)
        {
            case PositionType::Speaker:
                j = "s";
                break;
            case PositionType::Probe:
                j = "m";
                break;
            default:
                THROW_NOT_SUPPORTED_EXCEPTION("Not supported position type");
        }
    }

    inline void from_json(const nlohmann::json& j, PositionType& o)
    {
        if (j == "s")
        {
            o = PositionType::Speaker;
        }
        else if (j == "m")
        {
            o = PositionType::Probe;
        }
        else
        {
            THROW_NOT_SUPPORTED_EXCEPTION("Not supported position type");
        }
    }
}

#endif
