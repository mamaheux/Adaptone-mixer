#include <SignalProcessing/Parameters/DelayParameters.h>

using namespace adaptone;
using namespace std;

DelayParameters::DelayParameters(size_t channelCount, size_t maxDelay) :
    m_maxDelay(maxDelay), m_delays(channelCount, 0)
{
}

DelayParameters::~DelayParameters()
{
}
