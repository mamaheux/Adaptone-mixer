#include <Uniformization/Communication/ProbeDiscoverer.h>

#include <Uniformization/Communication/BoostAsioUtils.h>
#include <Uniformization/Communication/Messages/Udp/ProbeDiscoveryRequestMessage.h>
#include <Uniformization/Communication/Messages/Udp/ProbeDiscoveryResponseMessage.h>
#include <Uniformization/Communication/Messages/Udp/UdpMessageUtils.h>

#include <Utils/Exception/NetworkException.h>

using namespace adaptone;
using namespace std;

DiscoveredProbe::DiscoveredProbe(const boost::asio::ip::address& address) : m_address(address)
{
}

DiscoveredProbe::~DiscoveredProbe()
{
}

ProbeDiscoverer::ProbeDiscoverer(const Endpoint& endpoint, int timeoutMs, size_t discoveryTrialCount) :
    m_discoveryTrialCount(discoveryTrialCount),
    m_sendingBuffer(MaxNetworkBufferSize),
    m_receivingBuffer(MaxNetworkBufferSize)
{
    boost::system::error_code error;

    boost::asio::ip::address ipAddress = boost::asio::ip::address::from_string(endpoint.ipAddress(), error);
    if (error || !ipAddress.is_v4())
    {
        THROW_NETWORK_EXCEPTION("Invalid ip v4 address");
    }

    m_endpoint = boost::asio::ip::udp::endpoint(ipAddress, endpoint.port());

    unique_ptr<boost::asio::io_service> ioService = make_unique<boost::asio::io_service>();
    unique_ptr<boost::asio::ip::udp::socket> socket = make_unique<boost::asio::ip::udp::socket>(*ioService);

    socket->open(boost::asio::ip::udp::v4(), error);
    if (error)
    {
        THROW_NETWORK_EXCEPTION("Socket opening error");
    }

    socket->set_option(boost::asio::socket_base::broadcast(true), error);
    if (error)
    {
        THROW_NETWORK_EXCEPTION("Unable to set the broadcast flag");
    }

    if (!setReceivingTimeout(*socket, timeoutMs))
    {
        THROW_NETWORK_EXCEPTION("Unable to set the receive timeout");
    }

    socket->bind(boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), endpoint.port()), error);
    if (error)
    {
        THROW_NETWORK_EXCEPTION("Unable to bind the port");
    }

    m_ioService = move(ioService);
    m_socket = move(socket);
}

ProbeDiscoverer::~ProbeDiscoverer()
{
}

vector<DiscoveredProbe> ProbeDiscoverer::discover()
{
    vector<DiscoveredProbe> discoveredProbes;
    ProbeDiscoveryRequestMessage discoveryRequestMessage;
    discoveryRequestMessage.toBuffer(m_sendingBuffer);

    boost::system::error_code error;
    boost::asio::ip::udp::endpoint probeEndpoint;

    for (size_t i = 0; i < m_discoveryTrialCount; i++)
    {
        m_socket->send_to(toAsioBuffer(m_sendingBuffer, discoveryRequestMessage.fullSize()), m_endpoint, 0, error);
        if (error)
        {
            THROW_NETWORK_EXCEPTION("Unable to send the discovery message request");
        }
        //Read the broadcasted request
        readUdpMessageOrNull<ProbeDiscoveryRequestMessage>(*m_socket, probeEndpoint, m_receivingBuffer);

        optional<ProbeDiscoveryResponseMessage> response;
        do
        {
            response = readUdpMessageOrNull<ProbeDiscoveryResponseMessage>(*m_socket, probeEndpoint, m_receivingBuffer);

            if (response != nullopt)
            {
                discoveredProbes.emplace_back(probeEndpoint.address());
            }
        } while (response != nullopt);
    }

    return discoveredProbes;
}
