#include <Uniformization/UniformizationService.h>

using namespace adaptone;
using namespace std;

UniformizationService::UniformizationService(shared_ptr<Logger> logger,
    shared_ptr<GenericSignalOverride> signalOverride,
    const UniformizationServiceParameters& parameters) :
    m_logger(logger),
    m_signalOverride(signalOverride),
    m_parameters(parameters)
{
    shared_ptr<UniformizationProbeMessageHandler> probeMessageHandler = make_shared<UniformizationProbeMessageHandler>(
        m_logger, m_signalOverride->getSignalOverride<HeadphoneProbeSignalOverride>());
    unique_ptr<ProbeServers> probeServers = make_unique<ProbeServers>(m_logger, probeMessageHandler,
        parameters.toProbeServerParameters());

    m_probeMessageHandler = probeMessageHandler;
    m_probeServers = move(probeServers);
}

UniformizationService::~UniformizationService()
{
    stop();
}

void UniformizationService::start()
{
    m_probeServers->start();
}

void UniformizationService::stop()
{
    m_probeServers->stop();
}

void UniformizationService::listenToProbeSound(size_t probeId)
{
    m_signalOverride->getSignalOverride<HeadphoneProbeSignalOverride>()->setCurrentProbeId(probeId);
    m_signalOverride->setCurrentSignalOverrideType<HeadphoneProbeSignalOverride>();
}
