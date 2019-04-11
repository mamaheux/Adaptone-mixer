#ifndef MIXER_MIXER_APPLICATION_MESSAGE_HANDLER_H
#define MIXER_MIXER_APPLICATION_MESSAGE_HANDLER_H

#include <Communication/Handlers/ApplicationMessageHandler.h>

#include <SignalProcessing/SignalProcessor.h>

#include <memory>

namespace adaptone
{
    class MixerApplicationMessageHandler : public ApplicationMessageHandler
    {
        std::shared_ptr<SignalProcessor> m_signalProcessor;

    public:
        MixerApplicationMessageHandler(std::shared_ptr<SignalProcessor> signalProcessor);
        ~MixerApplicationMessageHandler() override;

        DECLARE_NOT_COPYABLE(MixerApplicationMessageHandler);
        DECLARE_NOT_MOVABLE(MixerApplicationMessageHandler);

        void handleDeserialized(const ApplicationMessage& message) override;
    };
}

#endif
