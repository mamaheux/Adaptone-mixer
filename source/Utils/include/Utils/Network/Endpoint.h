#ifndef UTILS_DATA_ENDPOINT_H
#define UTILS_DATA_ENDPOINT_H

#include <Utils/Configuration/Properties.h>

#include <string>
#include <cstdint>

namespace adaptone
{
    class Endpoint
    {
        std::string m_ipAddress;
        uint16_t m_port;

    public:
        Endpoint();
        Endpoint(const std::string& ipAddress, uint16_t port);
        virtual ~Endpoint();

        const std::string& ipAddress() const;
        uint16_t port() const;
    };

    inline const std::string& Endpoint::ipAddress() const
    {
        return m_ipAddress;
    }

    inline uint16_t Endpoint::port() const
    {
        return m_port;
    }

    template<>
    struct ValueParser<Endpoint>
    {
        static Endpoint parse(const std::string& key, const std::string& valueStr)
        {
            if (valueStr.size() < 3)
            {
                THROW_PROPERTY_PARSE_EXCEPTION(key, valueStr);
            }

            std::size_t colonPosition = valueStr.find_last_of(':');
            if (colonPosition == std::string::npos || colonPosition == valueStr.size() - 1)
            {
                THROW_PROPERTY_PARSE_EXCEPTION(key, valueStr);
            }

            uint16_t port = ValueParser<uint16_t>::parse(key + "Port", valueStr.substr(colonPosition + 1));

            if (valueStr[0] == '[')
            {
                std::size_t lastBracket = valueStr.find_last_of(']');
                if (lastBracket == std::string::npos)
                {
                    THROW_PROPERTY_PARSE_EXCEPTION(key, valueStr);
                }
                return Endpoint(valueStr.substr(1, lastBracket - 1), port);
            }

            return Endpoint(valueStr.substr(0, colonPosition), port);
        }
    };
}

#endif
