#ifndef UNIFORMIZATION_COMMUNICATION_PROBE_DISCOVERER_H
#define UNIFORMIZATION_COMMUNICATION_PROBE_DISCOVERER_H

#include <Utils/Network/Endpoint.h>
#include <Utils/Network/NetworkBuffer.h>

#include <boost/asio.hpp>

#include <memory>
#include <vector>

namespace adaptone
{
    class DiscoveredProbe
    {
        boost::asio::ip::address m_address;
    public:
        DiscoveredProbe(const boost::asio::ip::address& address);
        virtual ~DiscoveredProbe();

        const boost::asio::ip::address& address() const;
    };

    inline const boost::asio::ip::address& DiscoveredProbe::address() const
    {
        return m_address;
    }

    class ProbeDiscoverer
    {
        std::size_t m_discoveryTrialCount;
        NetworkBuffer m_sendingBuffer;
        NetworkBuffer m_receivingBuffer;

        boost::asio::ip::udp::endpoint m_endpoint;

        std::unique_ptr<boost::asio::io_service> m_ioService;
        std::unique_ptr<boost::asio::ip::udp::socket> m_socket;

    public:
        ProbeDiscoverer(const Endpoint& endpoint, int timeoutMs, std::size_t discoveryTrialCount);
        virtual ~ProbeDiscoverer();

        std::vector<DiscoveredProbe> discover();
    };
}

#endif
