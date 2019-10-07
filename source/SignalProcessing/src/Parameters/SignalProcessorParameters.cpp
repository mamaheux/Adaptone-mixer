#include <SignalProcessing/SignalProcessorParameters.h>

using namespace adaptone;
using namespace std;

SignalProcessorParameters::SignalProcessorParameters(ProcessingDataType processingDataType,
    size_t frameSampleCount,
    size_t sampleFrequency,
    size_t inputChannelCount,
    size_t outputChannelCount,
    PcmAudioFrameFormat inputFormat,
    PcmAudioFrameFormat outputFormat,
    const vector<double>& eqCenterFrequencies,
    size_t maxOutputDelay,
    size_t soundLevelLength) :
    m_processingDataType(processingDataType),
    m_frameSampleCount(frameSampleCount),
    m_sampleFrequency(sampleFrequency),
    m_inputChannelCount(inputChannelCount),
    m_outputChannelCount(outputChannelCount),
    m_inputFormat(inputFormat),
    m_outputFormat(outputFormat),
    m_eqCenterFrequencies(eqCenterFrequencies),
    m_maxOutputDelay(maxOutputDelay),
    m_soundLevelLength(soundLevelLength)
{
}

SignalProcessorParameters::~SignalProcessorParameters()
{
}
