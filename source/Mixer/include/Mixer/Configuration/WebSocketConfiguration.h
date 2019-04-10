#ifndef MIXER_CONFIGURATION_WEB_SOCKET_CONFIGURATION_H
#define MIXER_CONFIGURATION_WEB_SOCKET_CONFIGURATION_H

#include <Utils/Configuration/Properties.h>

#include <cstdint>
#include <string>

namespace adaptone
{
    class WebSocketConfiguration
    {
        std::string m_endpoint;
        uint16_t m_port;

    public:
        explicit WebSocketConfiguration(const Properties& properties);
        virtual ~WebSocketConfiguration();

        const std::string& endpoint() const;
        uint16_t port() const;
    };

    inline const std::string& WebSocketConfiguration::endpoint() const
    {
        return m_endpoint;
    }

    inline uint16_t WebSocketConfiguration::port() const
    {
        return m_port;
    }
}

#endif
