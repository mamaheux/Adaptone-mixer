#ifndef UNIFORMIZATION_UNIFORMIZATION_SERVICE_H
#define UNIFORMIZATION_UNIFORMIZATION_SERVICE_H

#include <Uniformization/UniformizationServiceParameters.h>
#include <Uniformization/UniformizationProbeMessageHandler.h>
#include <Uniformization/Communication/ProbeServers.h>
#include <Uniformization/SignalOverride/GenericSignalOverride.h>


#include <Utils/Logger/Logger.h>

#include <memory>

namespace adaptone
{
    class UniformizationService
    {
        std::shared_ptr<Logger> m_logger;
        std::shared_ptr<GenericSignalOverride> m_signalOverride;

        const UniformizationServiceParameters& m_parameters;

        std::shared_ptr<UniformizationProbeMessageHandler> m_probeMessageHandler;
        std::unique_ptr<ProbeServers> m_probeServers;

    public:
        UniformizationService(std::shared_ptr<Logger> logger,
            std::shared_ptr<GenericSignalOverride> signalOverride,
            const UniformizationServiceParameters& parameters);
        virtual ~UniformizationService();

        void start();
        void stop();

        void listenToProbeSound(std::size_t probeId);
    };
}

#endif
