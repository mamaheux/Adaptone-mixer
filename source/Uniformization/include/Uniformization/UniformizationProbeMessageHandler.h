#ifndef UNIFORMIZATION_UNIFORMIZATION_PROBE_MESSAGE_HANDLER_H
#define UNIFORMIZATION_UNIFORMIZATION_PROBE_MESSAGE_HANDLER_H

#include <Uniformization/Communication/ProbeMessageHandler.h>
#include <Uniformization/Communication/RecordResponseMessageAgregator.h>
#include <Uniformization/Communication/Messages/Udp/ProbeSoundDataMessage.h>
#include <Uniformization/Communication/Messages/Tcp/RecordResponseMessage.h>
#include <Uniformization/SignalOverride/HeadphoneProbeSignalOverride.h>

#include <Utils/Logger/Logger.h>

#include <functional>
#include <memory>
#include <unordered_map>

namespace adaptone
{
    class UniformizationProbeMessageHandler : public ProbeMessageHandler
    {
        std::shared_ptr<Logger> m_logger;
        std::shared_ptr<HeadphoneProbeSignalOverride> m_headphoneProbeSignalOverride;
        std::shared_ptr<RecordResponseMessageAgregator> m_recordResponseMessageAgregator;

        std::unordered_map<uint32_t, std::function<void(const ProbeMessage&, std::size_t, bool)>> m_handlersById;

    public:
        UniformizationProbeMessageHandler(std::shared_ptr<Logger> logger,
            std::shared_ptr<HeadphoneProbeSignalOverride> headphoneProbeSignalOverride,
            std::shared_ptr<RecordResponseMessageAgregator> recordResponseMessageAgregator);
        ~UniformizationProbeMessageHandler() override;

        void handle(const ProbeMessage& message, uint32_t probeId, bool isMaster) override;

    private:
        void handleProbeSoundDataMessage(const ProbeSoundDataMessage& message, uint32_t probeId, bool isMaster);
        void handleRecordResponseMessage(const RecordResponseMessage& message, uint32_t probeId, bool isMaster);
    };
}

#endif
