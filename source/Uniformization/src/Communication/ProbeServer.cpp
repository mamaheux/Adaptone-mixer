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
    shared_ptr<ProbeMessageHandler> messageHandler,
    const DiscoveredProbe& discoveredProbe,
    ProbeServerParameters probeServerParameters) :
    m_logger(move(logger)),
    m_messageHandler(move(messageHandler)),
    m_isMaster(false),
    m_isConnected(false),
    m_probeServerParameters(move(probeServerParameters)),
    m_sendingBuffer(MaxTcpMessageSize),
    m_stopped(true)
{
    m_endpoint = boost::asio::ip::tcp::endpoint(discoveredProbe.address(), m_probeServerParameters.tcpConnectionPort());

    unique_ptr<boost::asio::io_service> ioService = make_unique<boost::asio::io_service>();
    unique_ptr<boost::asio::ip::tcp::socket> socket = make_unique<boost::asio::ip::tcp::socket>(*ioService);

    connect(*socket);

    m_ioService = move(ioService);
    m_socket = move(socket);
}

ProbeServer::~ProbeServer()
{
    stop();
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
                updateHeartbeat();
                readMessage();
            }
            else
            {
                reconnect();
            }
        }
        catch (NetworkTimeoutException& ex)
        {
            m_logger->log(Logger::Level::Debug, ex);
        }
        catch (exception& ex)
        {
            m_isConnected.store(false);
            m_logger->log(Logger::Level::Error, ex);
        }
    }
}

void ProbeServer::connect(boost::asio::ip::tcp::socket& socket)
{
    boost::system::error_code error;
    socket.open(boost::asio::ip::tcp::v4(), error);
    if (error)
    {
        THROW_NETWORK_EXCEPTION("Socket opening error");
    }

    if (!setReceivingTimeout(socket, m_probeServerParameters.probeTimeoutMs()))
    {
        THROW_NETWORK_EXCEPTION("Unable to set the receive timeout");
    }

    socket.connect(m_endpoint, error);
    if (error)
    {
        THROW_NETWORK_EXCEPTION("Socket connection error");
    }

    ProbeInitializationRequestMessage request(m_probeServerParameters.sampleFrequency(),
        m_probeServerParameters.format());
    auto response = sendRequest<ProbeInitializationResponseMessage>(socket, m_sendingBuffer, request);
    if (!response.isCompatible())
    {
        THROW_NETWORK_EXCEPTION("Not compatible probe");
    }
    m_isMaster.store(response.isMaster());
    m_id.store(response.probeId());
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
    const chrono::system_clock::duration HeartbeatInterval =
        chrono::milliseconds(5 * m_probeServerParameters.probeTimeoutMs());
    const chrono::system_clock::duration HeartbeatTimeout =
        chrono::milliseconds(20 * m_probeServerParameters.probeTimeoutMs());

    auto now = chrono::system_clock::now();
    if (now - m_lastHeartbeatSendingTime > HeartbeatInterval)
    {
        send(HeartbeatMessage());
        m_lastHeartbeatSendingTime = now;
    }
    if (now - m_lastHeartbeatReceivingTime > HeartbeatTimeout)
    {
        m_isConnected.store(false);
    }
}
