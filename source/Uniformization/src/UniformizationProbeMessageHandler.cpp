#include <Uniformization/UniformizationProbeMessageHandler.h>

#include <Utils/Exception/NotSupportedException.h>

using namespace adaptone;
using namespace std;

#define ADD_HANDLE_FUNCTION(type) m_handlersById[type::Id] = [this](const ProbeMessage& message, size_t probeId, \
        bool isMaster) \
    { \
        handle##type(dynamic_cast<const type&>(message), probeId, isMaster);\
    }

UniformizationProbeMessageHandler::UniformizationProbeMessageHandler(shared_ptr<Logger> logger,
    shared_ptr<HeadphoneProbeSignalOverride> headphoneProbeSignalOverride) :
    m_logger(logger),
    m_headphoneProbeSignalOverride(headphoneProbeSignalOverride)
{
    ADD_HANDLE_FUNCTION(ProbeSoundDataMessage);
    ADD_HANDLE_FUNCTION(RecordResponseMessage);
    ADD_HANDLE_FUNCTION(FftResponseMessage);
}

UniformizationProbeMessageHandler::~UniformizationProbeMessageHandler()
{
}

void UniformizationProbeMessageHandler::handle(const ProbeMessage& message, size_t probeId, bool isMaster)
{
    auto it = m_handlersById.find(message.id());
    if (it == m_handlersById.end())
    {
        THROW_NOT_SUPPORTED_EXCEPTION("Not supported message");
    }
    it->second(message, probeId, isMaster);
}

void UniformizationProbeMessageHandler::handleProbeSoundDataMessage(const ProbeSoundDataMessage& message, size_t probeId,
    bool isMaster)
{
    m_headphoneProbeSignalOverride->writeData(message, probeId);
}

void UniformizationProbeMessageHandler::handleRecordResponseMessage(const RecordResponseMessage& message, size_t probeId,
    bool isMaster)
{
}

void UniformizationProbeMessageHandler::handleFftResponseMessage(const FftResponseMessage& message, size_t probeId,
    bool isMaster)
{
}
