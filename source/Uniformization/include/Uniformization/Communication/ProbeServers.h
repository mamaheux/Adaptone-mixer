#ifndef UNIFORMIZATION_COMMUNICATION_PROBE_SERVERS_H
#define UNIFORMIZATION_COMMUNICATION_PROBE_SERVERS_H

#include <Uniformization/Communication/ProbeServerParameters.h>
#include <Uniformization/Communication/ProbeDiscoverer.h>
#include <Uniformization/Communication/ProbeMessageHandler.h>
#include <Uniformization/Communication/ProbeServer.h>
#include <Uniformization/Communication/Messages/Udp/UdpMessageReader.h>

#include <Utils/ClassMacro.h>
#include <Utils/Logger/Logger.h>

#include <atomic>
#include <memory>
#include <thread>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>

namespace adaptone
{
    class ProbeServers
    {
        std::shared_ptr<Logger> m_logger;

        ProbeServerParameters m_probeServerParameters;

        std::shared_ptr<ProbeMessageHandler> m_messageHandler;

        ProbeDiscoverer m_probeDiscoverer;
        std::unordered_map<uint32_t, std::unique_ptr<ProbeServer>> m_probeServersById;
        std::map<boost::asio::ip::address, uint32_t> m_probeIdsByAddress;
        uint32_t m_masterProbeId;

        std::unique_ptr<boost::asio::io_service> m_ioService;
        std::unique_ptr<boost::asio::ip::udp::socket> m_udpSocket;
        UdpMessageReader m_udpMessageReader;

        std::atomic<bool> m_stopped;
        std::unique_ptr<std::thread> m_serverThread;
        std::shared_mutex m_mutex;

    public:
        ProbeServers(std::shared_ptr<Logger> logger,
            std::shared_ptr<ProbeMessageHandler> messageHandler,
            const ProbeServerParameters& probeServerParameters);
        virtual ~ProbeServers();

        DECLARE_NOT_COPYABLE(ProbeServers);
        DECLARE_NOT_MOVABLE(ProbeServers);

        void start();
        void stop();

        std::size_t probeCount();
        uint32_t masterProbeId();

        void sendToProbes(const ProbeMessage& message);

    private:
        void run();

        void createUdpSocket();
        void createProbeServer(const DiscoveredProbe& discoveredProbe);
        void readUdpSocket();
    };
}

#endif
