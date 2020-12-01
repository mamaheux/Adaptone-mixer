#include <Utils/Network/Endpoint.h>

using namespace adaptone;
using namespace std;

Endpoint::Endpoint() : m_ipAddress(""), m_port(0)
{
}

Endpoint::Endpoint(string ipAddress, uint16_t port) : m_ipAddress(move(ipAddress)), m_port(port)
{
}

Endpoint::~Endpoint()
{
}
