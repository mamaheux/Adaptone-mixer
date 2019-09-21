#include <Uniformization/Communication/ProbeServer.h>

#include <Uniformization/Communication/BoostAsioUtils.h>
#include <Uniformization/Communication/Messages/Tcp/TcpMessageReader.h>
#include <Uniformization/Communication/Messages/Tcp/ProbeInitializationRequestMessage.h>
#include <Uniformization/Communication/Messages/Tcp/ProbeInitializationResponseMessage.h>
#include <Uniformization/Communication/Messages/Tcp/HeartbeatMessage.h>

#include <Utils/Exception/NetworkException.h>

#include <chrono>

using namespace adaptone;
using namespace std;
using namespace std::chrono_literals;

ProbeServer::ProbeServer(shared_ptr<Logger> logger,
    uint16_t tcpConnectionPort,
    int timeoutMs,
    const DiscoveredProbe& discoveredProbe,
    size_t id,
    size_t sampleFrequency,
    PcmAudioFrame::Format format,
    shared_ptr<ProbeMessageHandler> messageHandler) :
    m_logger(logger),
    m_timeoutMs(timeoutMs),
    m_isMaster(false),
    m_isConnected(false),
    m_id(id),
    m_messageHandler(messageHandler),
    m_sendingBuffer(MaxTcpMessageSize)
{
    boost::asio::ip::tcp::endpoint endpoint(discoveredProbe.address(), tcpConnectionPort);

    unique_ptr<boost::asio::io_service> ioService = make_unique<boost::asio::io_service>();
    unique_ptr<boost::asio::ip::tcp::socket> socket = make_unique<boost::asio::ip::tcp::socket>(*ioService);

    boost::system::error_code error;
    socket->set_option(boost::asio::socket_base::broadcast(true), error);
    if (error)
    {
        THROW_NETWORK_EXCEPTION("Unable to set the broadcast flag");
    }

    if (!setReceivingTimeout(*socket, timeoutMs))
    {
        THROW_NETWORK_EXCEPTION("Unable to set the receive timeout");
    }

    m_socket->connect(endpoint, error);
    if (error)
    {
        THROW_NETWORK_EXCEPTION("Socket connection error");
    }

    ProbeInitializationRequestMessage request(sampleFrequency, format);
    auto response = sendRequest<ProbeInitializationResponseMessage>(*m_socket, m_sendingBuffer, request);
    if (!response.isCompatible())
    {
        THROW_NETWORK_EXCEPTION("Not compatible probe");
    }

    m_isMaster = response.isMaster();
    m_ioService = move(ioService);
    m_socket = move(socket);
    m_isConnected.store(true);
}

ProbeServer::~ProbeServer()
{
}

void ProbeServer::start()
{
    m_stopped.store(false);
    m_serverThread = make_unique<thread>(&ProbeServer::run, this);
}

void ProbeServer::stop()
{
    bool wasStopped = m_stopped.load();
    m_stopped.store(true);
    if (!wasStopped)
    {
        m_serverThread->join();
        m_serverThread.release();
    }
}

void ProbeServer::send(const ProbeMessage& message)
{
    lock_guard lock(m_mutex);
    if (m_isConnected.load())
    {
        boost::system::error_code error;

        message.toBuffer(m_sendingBuffer);
        m_socket->send(toAsioBuffer(m_sendingBuffer, message.fullSize()), 0, error);
        if (error)
        {
            THROW_NETWORK_EXCEPTION("Unable to send request");
        }
    }
    else
    {
        THROW_NETWORK_EXCEPTION("Probe not connected");
    }
}

void ProbeServer::run()
{
    auto lastHeartbeatSendingTime = chrono::system_clock::now();
    auto lastHeartbeatReceivingTime = chrono::system_clock::now();

    while (!m_stopped.load())
    {
        try
        {
            updateHeartbeat(lastHeartbeatSendingTime, lastHeartbeatReceivingTime);

            m_tcpMessageReader.read(*m_socket, [&](const ProbeMessage& message)
            {
                if (typeid(message) == typeid(HeartbeatMessage))
                {
                    lastHeartbeatReceivingTime = chrono::system_clock::now();
                }
                else
                {
                    m_messageHandler->handle(message, m_id);
                }
            });
        }
        catch (exception& ex)
        {
            m_logger->log(Logger::Level::Error, ex);
        }
    }
}

void ProbeServer::updateHeartbeat(std::chrono::system_clock::time_point& lastHeartbeatSendingTime,
    std::chrono::system_clock::time_point& lastHeartbeatReceivingTime)
{
    const chrono::system_clock::duration HeartbeatInterval = chrono::microseconds(10 * m_timeoutMs);
    const chrono::system_clock::duration HeartbeatTimeout = chrono::microseconds(20 * m_timeoutMs);

    auto now = chrono::system_clock::now();
    if (lastHeartbeatSendingTime - now > HeartbeatInterval)
    {
        send(HeartbeatMessage());
        lastHeartbeatSendingTime = now;
    }
    if (lastHeartbeatReceivingTime - now > HeartbeatTimeout)
    {
        m_isConnected.store(false);
    }
}
