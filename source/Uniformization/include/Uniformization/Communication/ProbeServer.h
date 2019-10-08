#ifndef UNIFORMIZATION_COMMUNICATION_PROBE_SERVER_H
#define UNIFORMIZATION_COMMUNICATION_PROBE_SERVER_H

#include <Uniformization/Communication/ProbeServerParameters.h>
#include <Uniformization/Communication/ProbeDiscoverer.h>
#include <Uniformization/Communication/ProbeMessageHandler.h>
#include <Uniformization/Communication/Messages/Tcp/TcpMessageReader.h>

#include <Utils/ClassMacro.h>
#include <Utils/Data/PcmAudioFrameFormat.h>
#include <Utils/Logger/Logger.h>

#include <boost/asio.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>

namespace adaptone
{
    class ProbeServer
    {
        std::shared_ptr<Logger> m_logger;
        std::shared_ptr<ProbeMessageHandler> m_messageHandler;
        ProbeServerParameters m_probeServerParameters;
        std::atomic<bool> m_isMaster;
        std::atomic<bool> m_isConnected;
        std::atomic<uint32_t> m_id;

        boost::asio::ip::tcp::endpoint m_endpoint;
        std::unique_ptr<boost::asio::io_service> m_ioService;
        std::unique_ptr<boost::asio::ip::tcp::socket> m_socket;
        NetworkBuffer m_sendingBuffer;
        TcpMessageReader m_tcpMessageReader;

        std::chrono::system_clock::time_point m_lastHeartbeatSendingTime;
        std::chrono::system_clock::time_point m_lastHeartbeatReceivingTime;

        std::atomic<bool> m_stopped;
        std::unique_ptr<std::thread> m_serverThread;
        std::mutex m_mutex;

    public:
        ProbeServer(std::shared_ptr<Logger> logger,
            std::shared_ptr<ProbeMessageHandler> messageHandler,
            const DiscoveredProbe& discoveredProbe,
            const ProbeServerParameters& probeServerParameters);
        virtual ~ProbeServer();

        DECLARE_NOT_COPYABLE(ProbeServer);
        DECLARE_NOT_MOVABLE(ProbeServer);

        void start();
        void stop();

        bool isMaster();
        bool isConnected();
        uint32_t id();

        void send(const ProbeMessage& message);

    private:
        void run();

        void connect(boost::asio::ip::tcp::socket& socket);
        void reconnect();
        void readMessage();
        void updateHeartbeat();
    };

    inline bool ProbeServer::isMaster()
    {
        return m_isMaster.load();
    }

    inline bool ProbeServer::isConnected()
    {
        return m_isConnected.load();
    }

    inline uint32_t ProbeServer::id()
    {
        return m_id.load();
    }
}

#endif
