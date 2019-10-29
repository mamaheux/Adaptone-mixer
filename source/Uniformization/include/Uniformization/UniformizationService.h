#ifndef UNIFORMIZATION_UNIFORMIZATION_SERVICE_H
#define UNIFORMIZATION_UNIFORMIZATION_SERVICE_H

#include <Uniformization/UniformizationServiceParameters.h>
#include <Uniformization/UniformizationProbeMessageHandler.h>
#include <Uniformization/Communication/ProbeServers.h>
#include <Uniformization/Communication/RecordResponseMessageAgregator.h>
#include <Uniformization/SignalOverride/GenericSignalOverride.h>
#include <Uniformization/SignalOverride/PassthroughSignalOverride.h>
#include <Uniformization/SignalOverride/SweepSignalOverride.h>
#include <Uniformization/Model/Room.h>

#include <SignalProcessing/SignalProcessor.h>

#include <Uniformization/Model/AutoPosition.h>
#include <Utils/ClassMacro.h>
#include <Utils/Logger/Logger.h>

#include <armadillo>

#include <memory>

namespace adaptone
{
    struct Metrics
    {
        arma::vec m_directivities;
        arma::vec m_delays;
    };

    class UniformizationService
    {
        std::shared_ptr<Logger> m_logger;
        std::shared_ptr<GenericSignalOverride> m_signalOverride;
        std::shared_ptr<SignalProcessor> m_signalProcessor;

        std::unique_ptr<AutoPosition> m_autoPosition;

        const UniformizationServiceParameters& m_parameters;

        std::shared_ptr<RecordResponseMessageAgregator> m_recordResponseMessageAgregator;
        std::shared_ptr<UniformizationProbeMessageHandler> m_probeMessageHandler;
        std::shared_ptr<ProbeServers> m_probeServers;

        std::atomic<bool> m_eqControlerEnabled;
        std::mutex m_probeServerMutex;

        std::atomic<bool> m_stopped;
        std::unique_ptr<std::thread> m_uniformizationThread;

        arma::mat m_speakersToProbesDistancesMat;
        Room m_room;

    public:
        UniformizationService(std::shared_ptr<Logger> logger,
            std::shared_ptr<GenericSignalOverride> signalOverride,
            std::shared_ptr<SignalProcessor> signalProcessor,
            const UniformizationServiceParameters& parameters);
        virtual ~UniformizationService();

        DECLARE_NOT_COPYABLE(UniformizationService);
        DECLARE_NOT_MOVABLE(UniformizationService);

        void start();
        void stop();

        void listenToProbeSound(uint32_t probeId);
        void stopProbeListening();

        Room initializeRoom(const std::vector<std::size_t>& masterOutputIndexes);
        void confirmRoomPositions();

    private:
        void run();

        void performEqControlIteration();

        void initializeRoomModelElementId(const std::vector<size_t>& masterOutputIndexes);
        std::unordered_map<uint32_t, AudioFrame<double>> sweepRoutineAtOutputX(const size_t masterOutputIndex);
        Metrics computeMetricsFromSweepData(std::unordered_map<uint32_t, AudioFrame<double>>& data);
        arma::mat distancesExtractionRoutine(const std::vector<size_t>& masterOutputIndexes);
    };
}

#endif
