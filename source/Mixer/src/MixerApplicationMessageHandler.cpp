#include <Mixer/MixerApplicationMessageHandler.h>

using namespace adaptone;
using namespace std;

MixerApplicationMessageHandler::MixerApplicationMessageHandler(shared_ptr<SignalProcessor> signalProcessor) :
    m_signalProcessor(signalProcessor)
{
}

MixerApplicationMessageHandler::~MixerApplicationMessageHandler()
{
}

void MixerApplicationMessageHandler::handleDeserialized(const ApplicationMessage& message)
{
}
