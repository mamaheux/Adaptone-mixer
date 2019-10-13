#include <Uniformization/Communication/ProbeServers.h>

#include <Uniformization/Communication/BoostAsioUtils.h>

#include <Utils/Exception/NetworkException.h>

using namespace adaptone;
using namespace std;

ProbeServers::ProbeServers(shared_ptr<Logger> logger,
    shared_ptr<ProbeMessageHandler> messageHandler,
    const ProbeServerParameters& probeServerParameters) :
    m_logger(logger),
    m_messageHandler(messageHandler),
    m_probeServerParameters(probeServerParameters),
    m_probeDiscoverer(probeServerParameters.discoveryEndpoint(),
        probeServerParameters.discoveryTimeoutMs(),
        probeServerParameters.discoveryTrialCount()),
    m_masterProbeId(-1),
    m_stopped(true)
{
}

ProbeServers::~ProbeServers()
{
    stop();
}

void ProbeServers::start()
{
    m_stopped.store(false);
    m_serverThread = make_unique<thread>(&ProbeServers::run, this);
}

void ProbeServers::stop()
{
    bool wasStopped = m_stopped.load();
    m_stopped.store(true);
    if (!wasStopped)
    {
        m_serverThread->join();
        m_serverThread.release();

        shared_lock lock(m_mutex);
        for (auto& pair : m_probeServersById)
        {
            pair.second->stop();
        }
        m_probeServersById.clear();
        m_probeIdsByAddress.clear();
    }
}

void ProbeServers::sendToProbes(const ProbeMessage& message)
{
    shared_lock lock(m_mutex);

    for (auto& pair : m_probeServersById)
    {
        pair.second->send(message);
    }
}

void ProbeServers::run()
{
    try
    {
        createUdpSocket();

        set<DiscoveredProbe> discoveredProbes = m_probeDiscoverer.discover();
        for (const DiscoveredProbe& discoveredProbe : discoveredProbes)
        {
            createProbeServer(discoveredProbe);
        }
        if (m_masterProbeId == -1)
        {
            THROW_NETWORK_EXCEPTION("No master probe found");
        }

        readUdpSocket();
    }
    catch (NetworkTimeoutException& ex)
    {
        m_logger->log(Logger::Level::Debug, ex);
    }
    catch (exception& ex)
    {
        m_logger->log(Logger::Level::Error, ex);
    }
}

void ProbeServers::createUdpSocket()
{
    boost::system::error_code error;
    m_ioService = make_unique<boost::asio::io_service>();
    m_udpSocket = make_unique<boost::asio::ip::udp::socket>(*m_ioService);

    m_udpSocket->open(boost::asio::ip::udp::v4(), error);
    if (error)
    {
        THROW_NETWORK_EXCEPTION("Socket opening error");
    }

    if (!setReceivingTimeout(*m_udpSocket, m_probeServerParameters.probeTimeoutMs()))
    {
        THROW_NETWORK_EXCEPTION("Unable to set the receive timeout");
    }

    m_udpSocket->bind(boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(),
        m_probeServerParameters.udpReceivingPort()), error);
    if (error)
    {
        THROW_NETWORK_EXCEPTION("Unable to bind the port");
    }
}

void ProbeServers::createProbeServer(const DiscoveredProbe& discoveredProbe)
{
    unique_lock lock(m_mutex);
    try
    {
        unique_ptr<ProbeServer> probeServer = make_unique<ProbeServer>(m_logger,
            m_messageHandler,
            discoveredProbe,
            m_probeServerParameters);

        uint32_t id = probeServer->id();
        m_probeServersById.emplace(id, move(probeServer));
        m_probeIdsByAddress.emplace(discoveredProbe.address(), id);
        m_probeServersById[id]->start();
        if (m_probeServersById[id]->isMaster())
        {
            m_masterProbeId = id;
        }
    }
    catch (exception& ex)
    {
        m_logger->log(Logger::Level::Error, ex);
    }
}

void ProbeServers::readUdpSocket()
{
    while (!m_stopped.load())
    {
        try
        {
            m_udpMessageReader.read(*m_udpSocket,
                [&](const ProbeMessage& message, const boost::asio::ip::address& address)
                {
                    uint32_t probeId = m_probeIdsByAddress[address];
                    m_messageHandler->handle(message, probeId, probeId == m_masterProbeId);
                });
        }
        catch (NetworkTimeoutException& ex)
        {
            m_logger->log(Logger::Level::Debug, ex);
        }
        catch (exception& ex)
        {
            m_logger->log(Logger::Level::Error, ex);
        }
    }
}
