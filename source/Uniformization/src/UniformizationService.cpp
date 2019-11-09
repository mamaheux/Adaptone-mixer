#include <Uniformization/UniformizationService.h>
#include <Uniformization/Communication/Messages/Tcp/RecordRequestMessage.h>
#include <Uniformization/Math.h>

#include <Utils/Exception/InvalidValueException.h>

#include <ctime>

#include <iostream>

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

    cout << " > > > m_parameters.sweepDuration() = " << m_parameters.sweepDuration() << endl;
    cout << " > > > m_parameters.sweepMaxDelay() = " << m_parameters.sweepMaxDelay() << endl;
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

    cout << "Room creation... ";
    m_room = Room(masterOutputIndexes.size(), m_probeServers->probeCount());
    cout << "Done!" << endl;

    cout << "Distance extraction routine : " << endl;
    m_speakersToProbesDistancesMat = distancesExtractionRoutine(masterOutputIndexes);
    cout << "Distance extraction routine : Done!" << endl;

    cout << "Compute room configuration : " << endl;
    m_autoPosition->computeRoomConfiguration2D(m_room, m_speakersToProbesDistancesMat, true);
    cout << "Compute room configuration : Done!" << endl;

    return m_room;
}

void UniformizationService::confirmRoomPositions()
{
    lock_guard lock(m_probeServerMutex);
    m_eqControlerEnabled.store(true);

    optimizeDelays();
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

void UniformizationService::initializeRoomModelElementId(const vector<size_t>& masterOutputIndexes)
{
    cout << " > Initializing room model element id... ";
    m_room.setSpeakersId(vector<uint32_t>(masterOutputIndexes.begin(), masterOutputIndexes.end()));
    m_room.setProbesId(m_probeServers->probeIds());
    cout << "Done!";
}

arma::mat UniformizationService::distancesExtractionRoutine(const vector<size_t>& masterOutputIndexes)
{
    initializeRoomModelElementId(masterOutputIndexes);

    std::vector<Speaker> speakers = m_room.speakers();
    arma::mat delaysMat = arma::zeros(masterOutputIndexes.size(), m_probeServers->probeCount());
    cout << " > Sweep main loop : " << endl;
    for (int i = 0; i < masterOutputIndexes.size(); i++)
    {
        cout << " > > Speaker index = " << i << endl;
        cout << " > > Sweep routine at output : " << endl;
        auto result = sweepRoutineAtOutputX(masterOutputIndexes[i]);
        cout << " > > Sweep routine at output : Done!" << endl;

        cout << " > > Compute metrics from sweep data : " << endl;
        Metrics metrics = computeMetricsFromSweepData(result);
        cout << " > > Compute metrics from sweep data : Done!" << endl;

        delaysMat.row(i).print();
        metrics.m_delays.print();
        cout << " > > delays assignement" << endl;
        delaysMat.row(i) = metrics.m_delays.t();
        cout << " > > directivities assignement" << endl;
        m_room.setSpeakerDirectivities(i, metrics.m_directivities);
    }

    return delaysMat * m_parameters.speedOfSound();
}

Metrics UniformizationService::computeMetricsFromSweepData(unordered_map<uint32_t, AudioFrame<double>>& audioFrames)
{
    size_t probesCount = audioFrames.size();
    Metrics metrics;
    arma::vec delays = arma::zeros<arma::vec>(probesCount);
    arma::mat directivities = arma::zeros<arma::mat>(probesCount, m_parameters.eqCenterFrequencies().size());
    const arma::vec sweepVec = m_signalOverride->getSignalOverride<SweepSignalOverride>()->sweepVec();
    auto probeIds = m_probeServers->probeIds();
    size_t n = 0;

    cout << " > > > Metrics extraction main loop : " << endl;
    for (uint32_t probeId : probeIds)
    {
        cout << " > > > > Computing delay... ";
        arma::vec probeData(audioFrames.at(probeId).data(), audioFrames.at(probeId).size(), false, false);
        int sampleDelay = std::max(0, findDelay(probeData, sweepVec));

        cout << "Done!" << endl;

        delays(n) = sampleDelay / static_cast<double>(m_parameters.sampleFrequency());
        delays(n) -= m_parameters.outputHardwareDelay();

        cout << " > > > > Computing directivities... ";
        constexpr bool Normalized = false;
        cout << " > > > > > ProbeData size = " << probeData.size() << endl;
        cout << " > > > > > SampleDelay = " << sampleDelay << endl;
        cout << " > > > > > SampleDelay + sweepVec.size() = " << sampleDelay + sweepVec.size() << endl;
        cout << " > > > > > Diff = " << sweepVec.size() << endl;
        arma::vec bandAverage = averageFrequencyBand(probeData(arma::span(sampleDelay, sampleDelay + sweepVec.size())),
            arma::conv_to<arma::vec>::from(m_parameters.eqCenterFrequencies()), m_parameters.sampleFrequency(),
            Normalized);
        cout << "Done!" << endl;

        directivities.row(n) = (bandAverage + 20 * log10(1 / (m_parameters.speedOfSound() * delays(n)) + 0.0001)).t();
        n++;
    }

    metrics.m_delays = delays;
    metrics.m_directivities = directivities;
    return metrics;
}

unordered_map<uint32_t, AudioFrame<double>> UniformizationService::sweepRoutineAtOutputX(const size_t masterOutputIndex)
{
    m_signalOverride->setCurrentSignalOverrideType<SweepSignalOverride>();

    timespec ts;
    timespec_get(&ts, TIME_UTC);
    struct tm recordStartTime;

    gmtime_r(&ts.tv_sec, &recordStartTime);

    cout << " > > > Current time (sec) : " << ts.tv_sec << endl;

    constexpr size_t recordDelay = 1;
    recordStartTime.tm_sec += recordDelay;
    ts.tv_sec += recordDelay;
    mktime(&recordStartTime);

    cout << " > > > Record time (sec) : " << ts.tv_sec << endl;

    constexpr int SecToMs = 1000;
    cout << " > > > m_parameters.sweepDuration() = " << m_parameters.sweepDuration() << endl;
    cout << " > > > m_parameters.sweepMaxDelay() = " << m_parameters.sweepMaxDelay() << endl;

    uint16_t durationMs = round((m_parameters.sweepDuration() + m_parameters.sweepMaxDelay()) * SecToMs);

    cout << " > > > Record duration (msec) : " << durationMs << endl;
    // Request record to all connected probes
    cout << " > > > Reseting record respond message agregator... ";
    m_recordResponseMessageAgregator->reset(masterOutputIndex, m_probeServers->probeCount());
    cout << "Done!" << endl;

    constexpr size_t Millisecond = 0;
    RecordRequestMessage message(recordStartTime.tm_hour, recordStartTime.tm_min, recordStartTime.tm_sec, Millisecond,
        durationMs, masterOutputIndex);

    cout << " > > > Sending record request to probes... ";
    m_probeServers->sendToProbes(message);
    cout << "Done!" << endl;

    cout << " > > > Waiting until record time is arrived... " << endl;
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
    cout << " > > > Starting sweep signal override" << endl;

    cout << " > > > Waiting for probes response message... ";
    constexpr float WaitTimeFactor = 1.5;
    auto result = m_recordResponseMessageAgregator->read(round(WaitTimeFactor * durationMs));
    if (result == nullopt)
    {
        THROW_NETWORK_EXCEPTION("Reading probes sweep data timeout");
    }
    cout << "Done!" << endl;

    m_signalOverride->setCurrentSignalOverrideType<PassthroughSignalOverride>();

    return *result;
}

void UniformizationService::optimizeDelays()
{
    auto speakers = m_room.speakers();
    auto probes = m_room.probes();

    arma::mat directivities = arma::zeros<arma::mat>(speakers.size(), probes.size());

    // Flatten frequency dimension in directivities matrix with a mean operation
    for (size_t i = 0; i < speakers.size(); i++)
    {
        directivities.row(i) = arma::mean(speakers[i].directivities(), 0).t();
    }

    arma::vec delays = findOptimalDelays(m_speakersToProbesDistancesMat, directivities, m_parameters.speedOfSound());
    vector<size_t> sampleDelays = arma::conv_to<vector<size_t>>::from(round(delays * m_parameters.sampleFrequency()));

    m_signalProcessor->setOutputDelays(sampleDelays);
}

