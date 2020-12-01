#include <Mixer/MixerConnectionHandler.h>

#include <limits>

using namespace adaptone;
using namespace std;

MixerConnectionHandler::MixerConnectionHandler(shared_ptr<SignalProcessor> signalProcessor, size_t outputChannelCount) :
    m_signalProcessor(move(signalProcessor)),
    m_outputChannelCount(outputChannelCount)
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
    m_signalProcessor->setOutputGains(vector<double>(m_outputChannelCount, 0));
}
