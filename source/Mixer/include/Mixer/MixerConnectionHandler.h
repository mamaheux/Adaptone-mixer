#ifndef MIXER_MIXER_CONNECTION_HANDLER_H
#define MIXER_MIXER_CONNECTION_HANDLER_H

#include <Communication/Handlers/ConnectionHandler.h>

#include <SignalProcessing/SignalProcessor.h>

#include <memory>

namespace adaptone
{
    class MixerConnectionHandler : public ConnectionHandler
    {
        std::shared_ptr<SignalProcessor> m_signalProcessor;

    public:
        MixerConnectionHandler(std::shared_ptr<SignalProcessor> signalProcessor);
        ~MixerConnectionHandler() override;

        DECLARE_NOT_COPYABLE(MixerConnectionHandler);
        DECLARE_NOT_MOVABLE(MixerConnectionHandler);

        void handleConnection() override;
        void handleDisconnection() override;
    };
}

#endif
