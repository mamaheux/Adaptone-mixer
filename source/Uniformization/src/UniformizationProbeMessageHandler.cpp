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
    shared_ptr<HeadphoneProbeSignalOverride> headphoneProbeSignalOverride,
    shared_ptr<RecordResponseMessageAgregator> recordResponseMessageAgregator) :
    m_logger(move(logger)),
    m_headphoneProbeSignalOverride(move(headphoneProbeSignalOverride)),
    m_recordResponseMessageAgregator(move(recordResponseMessageAgregator))
{
    ADD_HANDLE_FUNCTION(ProbeSoundDataMessage);
    ADD_HANDLE_FUNCTION(RecordResponseMessage);
}

UniformizationProbeMessageHandler::~UniformizationProbeMessageHandler()
{
}

void UniformizationProbeMessageHandler::handle(const ProbeMessage& message, uint32_t probeId, bool isMaster)
{
    auto it = m_handlersById.find(message.id());
    if (it == m_handlersById.end())
    {
        THROW_NOT_SUPPORTED_EXCEPTION("Not supported message");
    }
    it->second(message, probeId, isMaster);
}

void UniformizationProbeMessageHandler::handleProbeSoundDataMessage(const ProbeSoundDataMessage& message, uint32_t probeId,
    bool isMaster)
{
    m_headphoneProbeSignalOverride->writeData(message, probeId);
}

void UniformizationProbeMessageHandler::handleRecordResponseMessage(const RecordResponseMessage& message, uint32_t probeId,
    bool isMaster)
{
    m_recordResponseMessageAgregator->agregate(message, probeId);
}
