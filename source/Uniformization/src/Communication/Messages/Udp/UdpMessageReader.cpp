#include <Uniformization/Communication/Messages/Udp/UdpMessageReader.h>

#include <Uniformization/Communication/Messages/Udp/ProbeDiscoveryRequestMessage.h>
#include <Uniformization/Communication/Messages/Udp/ProbeDiscoveryResponseMessage.h>
#include <Uniformization/Communication/Messages/Udp/ProbeSoundDataMessage.h>

using namespace adaptone;
using namespace std;

#define ADD_HANDLE_FUNCTION(type) m_handlersById[type::Id] = [&](size_t messageSize, \
        const boost::asio::ip::address& address, \
        function<void(const ProbeMessage&, const boost::asio::ip::address&)>& readCallback) \
    { \
        readCallback(type::fromBuffer(m_buffer, messageSize), address); \
    }

UdpMessageReader::UdpMessageReader() : m_buffer(MaxUdpBufferSize)
{
    ADD_HANDLE_FUNCTION(ProbeDiscoveryRequestMessage);
    ADD_HANDLE_FUNCTION(ProbeDiscoveryResponseMessage);
    ADD_HANDLE_FUNCTION(ProbeSoundDataMessage);
}

UdpMessageReader::~UdpMessageReader()
{
}

void UdpMessageReader::read(boost::asio::ip::udp::socket& socket,
    function<void(const ProbeMessage&, const boost::asio::ip::address&)> readCallback)
{
    boost::system::error_code error;
    boost::asio::ip::udp::endpoint endpoint;
    size_t messageSize = udpSocketReceiveTimeout(socket, toAsioBuffer(m_buffer), endpoint, 0, error);
    if (error)
    {
        THROW_NETWORK_EXCEPTION("Unable to read the socket");
    }
    if (messageSize == 0)
    {
        THROW_NETWORK_EXCEPTION("Timeout");
    }
    if (messageSize < sizeof(uint32_t))
    {
        THROW_NETWORK_EXCEPTION("Invalid id");
    }

    uint32_t id = boost::endian::native_to_big(*reinterpret_cast<uint32_t*>(m_buffer.data()));
    auto it = m_handlersById.find(id);
    if (it == m_handlersById.end())
    {
        THROW_NETWORK_EXCEPTION("Invalid id");
    }

    it->second(messageSize, endpoint.address(), readCallback);
}
