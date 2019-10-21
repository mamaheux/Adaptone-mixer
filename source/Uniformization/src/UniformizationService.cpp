#include <Uniformization/UniformizationService.h>

using namespace adaptone;
using namespace std;
using namespace std::chrono_literals;

constexpr chrono::milliseconds UniformizationServiceSleepDuration(10);

UniformizationService::UniformizationService(shared_ptr<Logger> logger,
    shared_ptr<GenericSignalOverride> signalOverride,
    shared_ptr<SignalProcessor> signalProcessor,
    const UniformizationServiceParameters& parameters) :
    m_logger(logger),
    m_signalOverride(signalOverride),
    m_signalProcessor(signalProcessor),
    m_parameters(parameters),
    m_eqControlerEnabled(false),
    m_stopped(true)
{
    shared_ptr<RecordResponseMessageAgregator> recordResponseMessageAgregator =
        make_shared<RecordResponseMessageAgregator>(parameters.format());

    shared_ptr<UniformizationProbeMessageHandler> probeMessageHandler =
        make_shared<UniformizationProbeMessageHandler>(m_logger,
        m_signalOverride->getSignalOverride<HeadphoneProbeSignalOverride>(),
        recordResponseMessageAgregator);

    shared_ptr<ProbeServers> probeServers = make_shared<ProbeServers>(m_logger,
        probeMessageHandler,
        parameters.toProbeServerParameters());

    m_recordResponseMessageAgregator = recordResponseMessageAgregator;
    m_probeMessageHandler = probeMessageHandler;
    m_probeServers = probeServers;
}

UniformizationService::~UniformizationService()
{
    stop();
}

void UniformizationService::start()
{
    m_stopped.store(false);
    m_uniformizationThread = make_unique<thread>(&UniformizationService::run, this);

    m_probeServers->start();
}

void UniformizationService::stop()
{
    bool wasStopped = m_stopped.load();
    m_stopped.store(true);

    if (!wasStopped)
    {
        m_uniformizationThread->join();
        m_uniformizationThread.release();

        m_probeServers->stop();
    }
}

void UniformizationService::listenToProbeSound(uint32_t probeId)
{
    m_signalOverride->getSignalOverride<HeadphoneProbeSignalOverride>()->setCurrentProbeId(probeId);
    m_signalOverride->setCurrentSignalOverrideType<HeadphoneProbeSignalOverride>();
}

void UniformizationService::initializeRoom()
{
    m_eqControlerEnabled.store(false);
    lock_guard lock(m_probeServerMutex);

    //TODO add initiazation code
}

void UniformizationService::confirmRoomPositions()
{
    lock_guard lock(m_probeServerMutex);
    m_eqControlerEnabled.store(true);

    //TODO add confirmation code
}

void UniformizationService::run()
{
    try
    {
        while (!m_stopped.load())
        {
            if (m_eqControlerEnabled.load())
            {
                performEqControlIteration();
            }
            else
            {
                this_thread::sleep_for(UniformizationServiceSleepDuration);
            }
        }
    }
    catch (exception& ex)
    {
        m_logger->log(Logger::Level::Error, ex);
    }
}

void UniformizationService::performEqControlIteration()
{
    lock_guard lock(m_probeServerMutex);

    //TODO add eq control code
}
