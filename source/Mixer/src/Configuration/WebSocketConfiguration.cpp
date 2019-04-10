#include <Mixer/Configuration/WebSocketConfiguration.h>

using namespace adaptone;
using namespace std;

WebSocketConfiguration::WebSocketConfiguration(const Properties& properties)
{
    constexpr const char* EndpointPropertyKey = "web_socket.endpoint";
    constexpr const char* PortPropertyKey = "web_socket.port";

    m_endpoint = properties.get<string>(EndpointPropertyKey);
    m_port = properties.get<uint16_t>(PortPropertyKey);
}

WebSocketConfiguration::~WebSocketConfiguration()
{
}
