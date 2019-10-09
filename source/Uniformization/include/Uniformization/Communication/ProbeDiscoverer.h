#ifndef UNIFORMIZATION_COMMUNICATION_PROBE_DISCOVERER_H
#define UNIFORMIZATION_COMMUNICATION_PROBE_DISCOVERER_H

#include <Utils/ClassMacro.h>
#include <Utils/Network/Endpoint.h>
#include <Utils/Network/NetworkBuffer.h>

#include <boost/asio.hpp>

#include <memory>
#include <set>

namespace adaptone
{
    class DiscoveredProbe
    {
        boost::asio::ip::address m_address;

    public:
        DiscoveredProbe(const boost::asio::ip::address& address);
        virtual ~DiscoveredProbe();

        const boost::asio::ip::address& address() const;

        friend bool operator<(const DiscoveredProbe& l, const DiscoveredProbe& r);
        friend bool operator==(const DiscoveredProbe& l, const DiscoveredProbe& r);
    };

    inline const boost::asio::ip::address& DiscoveredProbe::address() const
    {
        return m_address;
    }

    inline bool operator<(const DiscoveredProbe& l, const DiscoveredProbe& r)
    {
        return l.m_address < r.m_address;
    }

    inline bool operator==(const DiscoveredProbe& l, const DiscoveredProbe& r)
    {
        return l.m_address == r.m_address;
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

        DECLARE_NOT_COPYABLE(ProbeDiscoverer);
        DECLARE_NOT_MOVABLE(ProbeDiscoverer);

        std::set<DiscoveredProbe> discover();
    };
}

#endif
