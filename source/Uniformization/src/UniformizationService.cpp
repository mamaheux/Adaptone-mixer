#include <Uniformization/UniformizationService.h>
#include <Uniformization/Communication/Messages/Tcp/RecordRequestMessage.h>
#include <Uniformization/Math.h>

#include <Utils/Exception/InvalidValueException.h>

#include <ctime>

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
    m_stopped(true),
    m_room(Room(0,0))
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

    unique_ptr<AutoPosition> autoPosition = make_unique<AutoPosition>(
        m_parameters.autoPositionAlpha(),
        m_parameters.autoPositionEpsilonTotalDistanceError(),
        m_parameters.autoPositionEpsilonDeltaTotalDistanceError(),
        m_parameters.autoPositionDistanceRelativeError(),
        m_parameters.autoPositionIterationCount(),
        m_parameters.autoPositionThermalIterationCount(),
        m_parameters.autoPositionTryCount(),
        m_parameters.autoPositionCountThreshold());

    m_autoPosition = move(autoPosition);
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

void UniformizationService::stopProbeListening()
{
    m_signalOverride->setCurrentSignalOverrideType<PassthroughSignalOverride>();
}

Room UniformizationService::initializeRoom(const vector<size_t>& masterOutputIndexes)
{
    m_eqControlerEnabled.store(false);
    lock_guard lock(m_probeServerMutex);

    m_speakersToProbesDistancesMat = distancesExtractionRoutine(masterOutputIndexes);

    m_room = Room(masterOutputIndexes.size(), m_probeServers->probeCount());
    m_autoPosition->computeRoomConfiguration2D(m_room, m_speakersToProbesDistancesMat, true);

    return m_room;
}

arma::mat UniformizationService::distancesExtractionRoutine(const vector<size_t>& masterOutputIndexes)
{
    arma::mat delaysMat = arma::zeros(masterOutputIndexes.size(), m_probeServers->probeCount());
    for (int i = 0; i < masterOutputIndexes.size(); i++)
    {
        auto result = sweepRoutineAtOutputX(masterOutputIndexes[i]);
        delaysMat.row(i) = computeDelaysFromSweepData(result);
    }

    return delaysMat / m_parameters.speedOfSound();
}

unordered_map<uint32_t, AudioFrame<double>> UniformizationService::sweepRoutineAtOutputX(const size_t masterOutputIndex)
{
    m_signalOverride->setCurrentSignalOverrideType<SweepSignalOverride>();

    timespec ts;
    timespec_get(&ts, TIME_UTC);
    struct tm recordStartTime;

    gmtime_r(&ts.tv_sec, &recordStartTime);
    constexpr size_t recordDelay = 1;
    recordStartTime.tm_sec += recordDelay;
    ts.tv_sec += recordDelay;
    mktime(&recordStartTime);

    constexpr int SecToMs = 1000;
    uint16_t durationMs = round((m_parameters.sweepDuration() + m_parameters.sweepMaxDelay()) * SecToMs);

    // Request record to all connected probes
    m_recordResponseMessageAgregator->reset(masterOutputIndex, m_probeServers->probeCount());

    constexpr size_t Millisecond = 0;
    RecordRequestMessage message(recordStartTime.tm_hour, recordStartTime.tm_min, recordStartTime.tm_sec, Millisecond,
        durationMs, masterOutputIndex);

    m_probeServers->sendToProbes(message);

    // Wait until record time is met
    size_t recordThreshold = ts.tv_sec;
    do
    {
        timespec_get(&ts, TIME_UTC);
        gmtime_r(&ts.tv_sec, &recordStartTime);
    }
    while (ts.tv_sec < recordThreshold);

    // Start emitting the sweep at the desired output
    m_signalOverride->getSignalOverride<SweepSignalOverride>()->startSweep(masterOutputIndex);

    constexpr float WaitTimeFactor = 1.5;
    auto result = m_recordResponseMessageAgregator->read(round(WaitTimeFactor * durationMs));
    if (result == nullopt)
    {
        THROW_NETWORK_EXCEPTION("Reading probes sweep data timeout");
    }

    m_signalOverride->setCurrentSignalOverrideType<PassthroughSignalOverride>();

    return *result;
}

arma::vec UniformizationService::computeDelaysFromSweepData(std::unordered_map<uint32_t,
    AudioFrame<double>>& audioFrames)
{
    size_t probesCount = audioFrames.size();
    arma::vec delays = arma::zeros<arma::vec>(probesCount);
    const arma::vec sweepVec = m_signalOverride->getSignalOverride<SweepSignalOverride>()->sweepVec();

    for (uint32_t i = 0; i < probesCount; i++)
    {
        arma::vec probeData(audioFrames.at(i).data(), audioFrames.at(i).size(), false, false);
        size_t sampleDelay = findDelay(probeData, sweepVec);
        delays(i) = sampleDelay / static_cast<double>(m_parameters.sampleFrequency());
        delays(i) += m_parameters.outputHardwareDelay();
    }

    return delays;
}

void UniformizationService::confirmRoomPositions()
{
    lock_guard lock(m_probeServerMutex);
    m_eqControlerEnabled.store(true);
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
