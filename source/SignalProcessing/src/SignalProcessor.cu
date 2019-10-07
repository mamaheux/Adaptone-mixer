#include <SignalProcessing/SignalProcessor.h>

#include <SignalProcessing/Cuda/CudaSignalProcessor.h>

#include <Utils/Exception/NotSupportedException.h>
#include <Utils/Exception/InvalidValueException.h>

using namespace adaptone;
using namespace std;

SignalProcessor::SignalProcessor(shared_ptr<AnalysisDispatcher> analysisDispatcher,
    const SignalProcessorParameters& parameters)
{
    if (parameters.frameSampleCount() < 2)
    {
        THROW_INVALID_VALUE_EXCEPTION("Invalid frame sample count value.", "< 2");
    }

    if (parameters.processingDataType() == ProcessingDataType::Float)
    {
        m_specificSignalProcessor = make_unique<CudaSignalProcessor<float>>(analysisDispatcher, parameters);
    }
    else if (parameters.processingDataType() == ProcessingDataType::Double)
    {
        m_specificSignalProcessor = make_unique<CudaSignalProcessor<double>>(analysisDispatcher, parameters);
    }
    else
    {
        THROW_NOT_SUPPORTED_EXCEPTION("Not supported processing data type");
    }
}

SignalProcessor::~SignalProcessor()
{
}
