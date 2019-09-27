#include <SignalProcessing/SignalProcessor.h>

#ifndef USE_CUDA

using namespace adaptone;
using namespace std;

SignalProcessor::SignalProcessor(shared_ptr<AnalysisDispatcher> analysisDispatcher,
    const SignalProcessorParameters& parameters)
{
    m_specificSignalProcessor = make_unique<SpecificSignalProcessor>();
}

SignalProcessor::~SignalProcessor()
{
}

#endif
