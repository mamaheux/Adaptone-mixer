#include <Mixer/MixerConnectionHandler.h>

using namespace adaptone;
using namespace std;

MixerConnectionHandler::MixerConnectionHandler(shared_ptr<SignalProcessor> signalProcessor) :
    m_signalProcessor(signalProcessor)
{
}

MixerConnectionHandler::~MixerConnectionHandler()
{
}

void MixerConnectionHandler::handleConnection()
{
}

void MixerConnectionHandler::handleDisconnection()
{
}
