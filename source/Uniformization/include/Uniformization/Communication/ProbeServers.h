#ifndef UNIFORMIZATION_COMMUNICATION_PROBE_SERVERS_H
#define UNIFORMIZATION_COMMUNICATION_PROBE_SERVERS_H

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

namespace adaptone
{
    class ProbeServers
    {
        std::shared_ptr<Logger> m_logger;

        uint16_t m_tcpConnectionPort;
        uint16_t m_udpReceivingPort;
        int m_probeTimeoutMs;
        std::size_t m_sampleFrequency;
        PcmAudioFrame::Format m_format;

        std::shared_ptr<ProbeMessageHandler> m_messageHandler;

        ProbeDiscoverer m_probeDiscoverer;
        std::unordered_map<std::size_t, std::unique_ptr<ProbeServer>> m_probeServersById;
        std::map<boost::asio::ip::address, std::size_t> m_probeIdsByAddress;
        std::size_t m_masterProbeId;

        std::unique_ptr<boost::asio::io_service> m_ioService;
        std::unique_ptr<boost::asio::ip::udp::socket> m_udpSocket;
        UdpMessageReader m_udpMessageReader;

        std::atomic<bool> m_stopped;
        std::unique_ptr<std::thread> m_serverThread;

    public:
        ProbeServers(std::shared_ptr<Logger> logger,
            const Endpoint& discoveryEndpoint,
            int discoveryTimeoutMs,
            std::size_t discoveryTrialCount,
            uint16_t tcpConnectionPort,
            uint16_t udpReceivingPort,
            int probeTimeoutMs,
            std::size_t sampleFrequency,
            PcmAudioFrame::Format format,
            std::shared_ptr<ProbeMessageHandler> messageHandler);
        virtual ~ProbeServers();

        DECLARE_NOT_COPYABLE(ProbeServers);
        DECLARE_NOT_MOVABLE(ProbeServers);

        void start();
        void stop();

    private:
        void run();

        void createUdpSocket();
        void createProbeServer(const DiscoveredProbe& discoveredProbe);
        void readUdpSocket();
    };
}

#endif
