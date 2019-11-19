#include <Uniformization/UniformizationService.h>
#include <Uniformization/Communication/Messages/Tcp/RecordRequestMessage.h>
#include <Uniformization/Math.h>

#include <Utils/Time.h>
#include <Utils/Exception/InvalidValueException.h>

#include <ctime>

#include <iostream>

#include <fstream>

using namespace adaptone;
using namespace std;
using namespace std::chrono_literals;
using namespace arma;

constexpr chrono::milliseconds EqControlSleepDuration(10);
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

    m_recordIndex = 0;
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
    cout << "listenToProbeSound : " << probeId << endl;
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

    m_room = Room(masterOutputIndexes.size(), m_probeServers->probeCount());
    m_speakersToProbesDistancesMat = distancesExtractionRoutine(masterOutputIndexes);
    m_autoPosition->computeRoomConfiguration2D(m_room, m_speakersToProbesDistancesMat, true);

    m_outputEqGains = zeros<mat>(m_room.speakers().size(), m_parameters.eqCenterFrequencies().size());

    return m_room;
}

void UniformizationService::confirmRoomPositions()
{
    lock_guard lock(m_probeServerMutex);

    optimizeDelays();
    m_eqControlerEnabled.store(true);
}

void UniformizationService::enable()
{
    lock_guard lock(m_probeServerMutex);

    auto speakers = m_room.speakers();

    for (size_t i = 0; i < speakers.size(); i++)
    {
        m_signalProcessor->setOutputDelay(speakers[i].id(), m_optimalSampleDelays.at(i));
    }
    for (size_t i = 0; i < speakers.size(); i++)
    {
        vector<double> outputEqGainsNatural = conv_to<vector<double>>::from(exp10(m_outputEqGains.row(i) / 20));
        m_signalProcessor->setUniformizationGraphicEqGains(speakers[i].id(), outputEqGainsNatural);
    }

    m_eqControlerEnabled.store(true);
}

void UniformizationService::disable()
{
    m_eqControlerEnabled.store(false);
    lock_guard lock(m_probeServerMutex);

    auto speakers = m_room.speakers();

    for (size_t i = 0; i < speakers.size(); i++)
    {
        m_signalProcessor->setOutputDelay(speakers[i].id(), 0);
    }
    for (size_t i = 0; i < speakers.size(); i++)
    {
        m_signalProcessor->setUniformizationGraphicEqGains(speakers[i].id(),
            vector<double>(m_parameters.eqCenterFrequencies().size(), 1));
    }
}

void UniformizationService::run()
{
    while (!m_stopped.load())
    {
        try
        {
            if (m_eqControlerEnabled.load())
            {
                this_thread::sleep_for(EqControlSleepDuration);
                performEqControlIteration();
            }
            else
            {
                this_thread::sleep_for(UniformizationServiceSleepDuration);
            }
        }
        catch (exception& ex)
        {
            m_logger->log(Logger::Level::Error, ex);
        }
    }
}

void UniformizationService::performEqControlIteration()
{
    lock_guard lock(m_probeServerMutex);
    if (m_eqControlerEnabled.load())
    {
        eqControl();
    }
}

void UniformizationService::initializeRoomModelElementId(const vector<size_t>& masterOutputIndexes)
{
    m_room.setSpeakersId(vector<uint32_t>(masterOutputIndexes.begin(), masterOutputIndexes.end()));
    m_room.setProbesId(m_probeServers->probeIds());
}

mat UniformizationService::distancesExtractionRoutine(const vector<size_t>& masterOutputIndexes)
{
    initializeRoomModelElementId(masterOutputIndexes);

    std::vector<Speaker> speakers = m_room.speakers();
    mat delaysMat = zeros(masterOutputIndexes.size(), m_probeServers->probeCount());
    for (int i = 0; i < masterOutputIndexes.size(); i++)
    {
        auto result = sweepRoutineAtOutputX(masterOutputIndexes[i]);
        Metrics metrics = computeMetricsFromSweepData(result);

        delaysMat.row(i) = metrics.m_delays.t();
        m_room.setSpeakerDirectivities(i, metrics.m_directivities);
    }

    for (uint32_t probeId : m_probeServers->probeIds())
    {
        cout << "Probe id  : " << probeId << endl;
    }

    cout << " > > delay matrix : " << endl;
    delaysMat.print();

    cout << " > > distance matrix : " << endl;
    (delaysMat * m_parameters.speedOfSound()).print();

    return delaysMat * m_parameters.speedOfSound();
}

Metrics UniformizationService::computeMetricsFromSweepData(unordered_map<uint32_t, AudioFrame<double>>& audioFrames)
{
    size_t probesCount = audioFrames.size();
    Metrics metrics;
    vec delays = zeros<vec>(probesCount);
    mat directivities = zeros<mat>(probesCount, m_parameters.eqCenterFrequencies().size());
    const vec sweepVec = m_signalOverride->getSignalOverride<SweepSignalOverride>()->sweepVec();
    auto probeIds = m_probeServers->probeIds();
    size_t n = 0;

    for (uint32_t probeId : probeIds)
    {
        vec probeData(audioFrames.at(probeId).data(), audioFrames.at(probeId).size(), false, false);

        cout << " > > Stats probe " << probeId << endl;
        cout << " > > > mean " << mean(abs(probeData)) << endl;
        cout << " > > > max " << max(abs(probeData)) << endl;
        
        size_t sampleDelay = max(static_cast<int64_t>(0), findDelay(probeData, sweepVec));

        delays(n) = sampleDelay / static_cast<double>(m_parameters.sampleFrequency());
        delays(n) -= m_parameters.outputHardwareDelay();
        delays(n) = max(0.0001, delays(n));

        constexpr bool Normalized = false;
        vec bandAverage = averageFrequencyBand(probeData(span(sampleDelay, sampleDelay + sweepVec.size())),
            conv_to<vec>::from(m_parameters.eqCenterFrequencies()), m_parameters.sampleFrequency(),
            Normalized);

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

    constexpr size_t SecToMs = 1000;
    uint16_t durationMs = round((m_parameters.sweepDuration() + m_parameters.sweepMaxDelay()) * SecToMs);

    constexpr size_t RecordDelayMs = 1000;
    timespec recordThreshold = sendProbesRecordRequestMessageNow(RecordDelayMs, durationMs, masterOutputIndex);

    waitUntilTimeReached(recordThreshold);

    timespec ts;
    timespec_get(&ts, TIME_UTC);

    // Start emitting the sweep at the desired output
    m_signalOverride->getSignalOverride<SweepSignalOverride>()->startSweep(masterOutputIndex);

    constexpr double durationFactor = 3.0;
    auto result = agregateProbesRecordResponseMessageNow(round(durationFactor * durationMs));

    m_signalOverride->setCurrentSignalOverrideType<PassthroughSignalOverride>();

    return result;
}

timespec UniformizationService::sendProbesRecordRequestMessageNow(size_t delayMs, size_t durationMs, size_t recordIndex)
{
    timespec ts;
    timespec_get(&ts, TIME_UTC);
    struct tm recordStartTime;

    ts = addMsToTimespec(delayMs, ts);
    gmtime_r(&ts.tv_sec, &recordStartTime);

    constexpr size_t MsToNs = 1000000;
    size_t millisecond = ts.tv_nsec / MsToNs;

    // Request record to all connected probes
    m_recordResponseMessageAgregator->reset(recordIndex, m_probeServers->probeCount());
    RecordRequestMessage message(recordStartTime.tm_hour, recordStartTime.tm_min, recordStartTime.tm_sec, millisecond,
        durationMs, recordIndex);
    m_probeServers->sendToProbes(message);

    return ts;
}

unordered_map<uint32_t, AudioFrame<double>> UniformizationService::agregateProbesRecordResponseMessageNow(size_t timeoutMs)
{
    auto result = m_recordResponseMessageAgregator->read(timeoutMs);
    if (result == nullopt)
    {
        THROW_NETWORK_EXCEPTION("Reading probes sweep data timeout");
    }

    return *result;
}

void UniformizationService::optimizeDelays()
{
    auto speakers = m_room.speakers();
    auto probes = m_room.probes();

    mat directivities = zeros<mat>(speakers.size(), probes.size());

    // Flatten frequency dimension in directivities matrix with a mean operation
    for (size_t i = 0; i < speakers.size(); i++)
    {
        directivities.row(i) = mean(speakers[i].directivities(), 1).t();
    }

    cout << " > directivities : " << endl;
    directivities.print();

    directivities = zeros<mat>(speakers.size(), probes.size());
    // DEBUG #############################################################
    vec delays = findOptimalDelays(m_speakersToProbesDistancesMat, exp10(directivities / 20), m_parameters.speedOfSound());
    m_optimalSampleDelays = conv_to<vector<size_t>>::from(round(delays * m_parameters.sampleFrequency()));

    cout << " > Speaker delays  : " << endl;
    delays.print();

    for (size_t i = 0; i < speakers.size(); i++)
    {
        m_signalProcessor->setOutputDelay(speakers[i].id(), m_optimalSampleDelays[i]);
    }
}

void UniformizationService::eqControl()
{
    m_recordIndex++;

    constexpr size_t RecordDelayMs = 1000;
    constexpr size_t SecToMs = 1000;
    size_t recordDurationMs = m_parameters.eqControlBlockSize() / static_cast<double>(m_parameters.sampleFrequency()) *
                              SecToMs;
    sendProbesRecordRequestMessageNow(RecordDelayMs, recordDurationMs, m_recordIndex);

    constexpr double durationFactor = 2;
    size_t timeoutMs = round(durationFactor * (recordDurationMs + RecordDelayMs));
    auto audioFrames = agregateProbesRecordResponseMessageNow(timeoutMs);

    vector<vec> bandAverageVector;
    vec targetBandAverage;

    computeBandAveragesFromAudioFrames(audioFrames, bandAverageVector, targetBandAverage);
    updateOutputEqGains(bandAverageVector, targetBandAverage);
}

void UniformizationService::computeBandAveragesFromAudioFrames(unordered_map<uint32_t, AudioFrame<double>>& audioFrames,
    vector<vec>& bandAverageVector, vec& targetBandAverage)
{
    size_t masterProbeId = m_probeServers->masterProbeId();
    auto probeIds = m_probeServers->probeIds();

    for (uint32_t probeId : probeIds)
    {
        vec probeData(audioFrames.at(probeId).data(), audioFrames.at(probeId).size(), false, false);

        constexpr bool Normalized = true;
        bandAverageVector.emplace_back(averageFrequencyBand(probeData,
            conv_to<vec>::from(m_parameters.eqCenterFrequencies()), m_parameters.sampleFrequency(), Normalized));

        cout << "probe bandAverage : " << probeId << endl;
        bandAverageVector.back().print();

        if (probeId == masterProbeId)
        {
            targetBandAverage = bandAverageVector.back();
        }
    }
}

void UniformizationService::updateOutputEqGains(vector<vec> bandAverageVector, vec targetBandAverage)
{
    constexpr size_t errorWindowSize = 20;
    static vector<rowvec> bandErrorTotalBuffer;

    size_t masterProbeId = m_probeServers->masterProbeId();

    auto speakers = m_room.speakers();
    auto probes = m_room.probes();
    mat normalizedDistances = m_speakersToProbesDistancesMat / max(vectorise(m_speakersToProbesDistancesMat));

    rowvec bandErrorTotal = zeros<rowvec>(m_parameters.eqCenterFrequencies().size());
    for (size_t i = 0; i < speakers.size(); i++)
    {
        mat directivities = exp10(speakers[i].directivities() / 20);
        directivities /= max(vectorise(directivities));
        directivities = ones<mat>(probes.size(), m_parameters.eqCenterFrequencies().size()); // DEBUG #############################################################

        rowvec bandError = zeros<rowvec>(m_parameters.eqCenterFrequencies().size());
        for (size_t j = 0; j < probes.size(); j++)
        {
            if (probes[j].id() != masterProbeId)
            {
                bandError +=
                    (directivities.row(j) / normalizedDistances(i, j)) % (targetBandAverage - bandAverageVector[j]).t();
            }
        }
        bandError /= probes.size();
        bandErrorTotal += abs(bandError);

        double eqCenterCorrection = -m_parameters.eqControlErrorCenterCorrectionFactor() * mean(m_outputEqGains.row(i));
        rowvec eqCorrection = clamp(m_parameters.eqControlErrorCorrectionFactor() * bandError,
            m_parameters.eqControlErrorCorrectionLowerBound(), m_parameters.eqControlErrorCorrectionUpperBound());

        m_outputEqGains.row(i) = clamp(m_outputEqGains.row(i) + eqCorrection + eqCenterCorrection,
            m_parameters.eqControlEqGainLowerBoundDb(), m_parameters.eqControlEqGainUpperBoundDb());

        vector<double> outputEqGainsNatural = conv_to<vector<double>>::from(exp10(m_outputEqGains.row(i) / 20));
        m_signalProcessor->setUniformizationGraphicEqGains(speakers[i].id(), outputEqGainsNatural);
    }

    bandErrorTotal /= speakers.size();
    bandErrorTotalBuffer.emplace_back(bandErrorTotal);
    if (bandErrorTotalBuffer.size() > errorWindowSize)
    {
        bandErrorTotalBuffer.erase(bandErrorTotalBuffer.begin());
    }

    bandErrorTotal = zeros<rowvec>(m_parameters.eqCenterFrequencies().size());
    for (size_t i = 0; i < bandErrorTotalBuffer.size(); i++)
    {
        bandErrorTotal += bandErrorTotalBuffer[i];
    }
    bandErrorTotal /= bandErrorTotalBuffer.size();
    cout << "BandErrorTotal mean : " << sum(bandErrorTotal) << endl;
}
