#include <SignalProcessing/Cuda/CudaSignalProcessor.h>

#include <SignalProcessing/SignalProcessor.h>

#include <iostream>

using namespace adaptone;
using namespace std;

CudaSignalProcessor::CudaSignalProcessor(ProcessingDataType processingDataType,
    size_t frameSampleCount,
    size_t sampleFrequency,
    size_t inputChannelCount,
    size_t outputChannelCount,
    PcmAudioFrame::Format inputFormat,
    PcmAudioFrame::Format outputFormat) :
    m_processingDataType(processingDataType),
    m_frameSampleCount(frameSampleCount),
    m_sampleFrequency(sampleFrequency),
    m_inputChannelCount(inputChannelCount),
    m_outputChannelCount(outputChannelCount),
    m_inputFormat(inputFormat),
    m_outputFormat(outputFormat)
{
}

CudaSignalProcessor::~CudaSignalProcessor()
{
}

const PcmAudioFrame& CudaSignalProcessor::process(const PcmAudioFrame& inputFrame)
{
    return inputFrame;
}
