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
    PcmAudioFrameFormat format,
    shared_ptr<ProbeMessageHandler> messageHandler) :
    m_logger(logger),
    m_timeoutMs(timeoutMs),
    m_isMaster(false),
    m_isConnected(false),
    m_id(id),
    m_sampleFrequency(sampleFrequency),
    m_format(format),
    m_messageHandler(messageHandler),
    m_sendingBuffer(MaxTcpMessageSize)
{
    m_endpoint = boost::asio::ip::tcp::endpoint(discoveredProbe.address(), tcpConnectionPort);

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

    connect(*socket);

    m_ioService = move(ioService);
    m_socket = move(socket);
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
        if (m_socket->is_open())
        {
            m_socket->close();
        }
        m_isConnected.store(false);
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
    while (!m_stopped.load())
    {
        try
        {
            if (m_isConnected.load())
            {
                readMessage();
                updateHeartbeat();
            }
            else
            {
                reconnect();
            }
        }
        catch (exception& ex)
        {
            m_logger->log(Logger::Level::Error, ex);
        }
    }
}

void ProbeServer::connect(boost::asio::ip::tcp::socket& socket)
{
    boost::system::error_code error;
    socket.connect(m_endpoint, error);
    if (error)
    {
        THROW_NETWORK_EXCEPTION("Socket connection error");
    }

    ProbeInitializationRequestMessage request(m_sampleFrequency, m_format);
    auto response = sendRequest<ProbeInitializationResponseMessage>(socket, m_sendingBuffer, request);
    if (!response.isCompatible())
    {
        THROW_NETWORK_EXCEPTION("Not compatible probe");
    }
    m_isMaster = response.isMaster();
    m_isConnected.store(true);

    m_lastHeartbeatSendingTime = chrono::system_clock::now();
    m_lastHeartbeatReceivingTime = chrono::system_clock::now();
}

void ProbeServer::reconnect()
{
    const chrono::system_clock::duration ConnectionInterval = 1s;
    this_thread::sleep_for(ConnectionInterval);

    if (m_socket->is_open())
    {
        m_socket->close();
    }
    connect(*m_socket);
}

void ProbeServer::readMessage()
{
    m_tcpMessageReader.read(*m_socket, [&](const ProbeMessage& message)
    {
        if (typeid(message) == typeid(HeartbeatMessage))
        {
            m_lastHeartbeatReceivingTime = chrono::system_clock::now();
        }
        else
        {
            m_messageHandler->handle(message, m_id, m_isMaster);
        }
    });
}

void ProbeServer::updateHeartbeat()
{
    const chrono::system_clock::duration HeartbeatInterval = chrono::microseconds(10 * m_timeoutMs);
    const chrono::system_clock::duration HeartbeatTimeout = chrono::microseconds(20 * m_timeoutMs);

    auto now = chrono::system_clock::now();
    if (m_lastHeartbeatSendingTime - now > HeartbeatInterval)
    {
        send(HeartbeatMessage());
        m_lastHeartbeatSendingTime = now;
    }
    if (m_lastHeartbeatReceivingTime - now > HeartbeatTimeout)
    {
        m_isConnected.store(false);
    }
}
