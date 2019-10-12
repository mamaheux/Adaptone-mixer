#ifndef UNIFORMIZATION_UNIFORMIZATION_SERVICE_H
#define UNIFORMIZATION_UNIFORMIZATION_SERVICE_H

#include <Uniformization/UniformizationServiceParameters.h>
#include <Uniformization/UniformizationProbeMessageHandler.h>
#include <Uniformization/Communication/ProbeServers.h>
#include <Uniformization/SignalOverride/GenericSignalOverride.h>

#include <SignalProcessing/SignalProcessor.h>

#include <Utils/ClassMacro.h>
#include <Utils/Logger/Logger.h>

#include <memory>

namespace adaptone
{
    class UniformizationService
    {
        std::shared_ptr<Logger> m_logger;
        std::shared_ptr<GenericSignalOverride> m_signalOverride;
        std::shared_ptr<SignalProcessor> m_signalProcessor;

        const UniformizationServiceParameters& m_parameters;

        std::shared_ptr<UniformizationProbeMessageHandler> m_probeMessageHandler;
        std::shared_ptr<ProbeServers> m_probeServers;

        std::atomic<bool> m_eqControlerEnabled;
        std::mutex m_probeServerMutex;

        std::atomic<bool> m_stopped;
        std::unique_ptr<std::thread> m_uniformizationThread;

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

        void listenToProbeSound(std::size_t probeId);
        void initializeRoom();
        void confirmRoomPositions();

    private:
        void run();

        void performEqControlIteration();
    };
}

#endif
