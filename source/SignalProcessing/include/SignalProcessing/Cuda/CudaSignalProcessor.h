#ifndef SIGNAL_PROCESSING_CUDA_CUDA_SIGNAL_PROCESSOR_H
#define SIGNAL_PROCESSING_CUDA_CUDA_SIGNAL_PROCESSOR_H

#include <SignalProcessing/ProcessingDataType.h>
#include <SignalProcessing/SpecificSignalProcessor.h>
#include <SignalProcessing/Cuda/CudaSignalProcessorBuffers.h>

#include <Utils/ClassMacro.h>
#include <Utils/Data/PcmAudioFrame.h>

#include <cuda_runtime.h>

namespace adaptone
{
    constexpr std::size_t CudaSignalProcessorFrameCount = 2;

    template<class T>
    class CudaSignalProcessor : public SpecificSignalProcessor
    {
        std::size_t m_frameSampleCount;
        std::size_t m_sampleFrequency;

        std::size_t m_inputChannelCount;
        std::size_t m_outputChannelCount;

        PcmAudioFrame::Format m_inputFormat;
        PcmAudioFrame::Format m_outputFormat;

        PcmAudioFrame m_outputFrame;
        CudaSignalProcessorBuffers<T> m_buffers;

    public:
        CudaSignalProcessor(std::size_t frameSampleCount,
            std::size_t sampleFrequency,
            std::size_t inputChannelCount,
            std::size_t outputChannelCount,
            PcmAudioFrame::Format inputFormat,
            PcmAudioFrame::Format outputFormat);
        virtual ~CudaSignalProcessor();

        DECLARE_NOT_COPYABLE(CudaSignalProcessor);
        DECLARE_NOT_MOVABLE(CudaSignalProcessor);

        const PcmAudioFrame& process(const PcmAudioFrame& inputFrame) override;
    };

    template<class T>
    CudaSignalProcessor<T>::CudaSignalProcessor(size_t frameSampleCount,
        size_t sampleFrequency,
        size_t inputChannelCount,
        size_t outputChannelCount,
        PcmAudioFrame::Format inputFormat,
        PcmAudioFrame::Format outputFormat) :
        m_frameSampleCount(frameSampleCount),
        m_sampleFrequency(sampleFrequency),
        m_inputChannelCount(inputChannelCount),
        m_outputChannelCount(outputChannelCount),
        m_inputFormat(inputFormat),
        m_outputFormat(outputFormat),
        m_outputFrame(outputFormat, outputChannelCount, frameSampleCount),
        m_buffers(PcmAudioFrame::size(inputFormat, inputChannelCount, frameSampleCount),
            PcmAudioFrame::size(outputFormat, outputChannelCount, frameSampleCount),
            CudaSignalProcessorFrameCount)
    {
    }

    template<class T>
    CudaSignalProcessor<T>::~CudaSignalProcessor()
    {
    }

    template<class T>
    __global__ void processKernel(CudaSignalProcessorBuffers<T> buffers)
    {
        int index = threadIdx.x;
        int stride = blockDim.x;

        uint8_t* inputFrame = buffers.currentInputFrame();
        uint8_t* outputFrame = buffers.currentOutputFrame();

        for (int i = index; i < buffers.inputFrameSize() && i < buffers.outputFrameSize(); i += stride)
        {
            outputFrame[i] = inputFrame[i];
        }
    }

    template<class T>
    const PcmAudioFrame& CudaSignalProcessor<T>::process(const PcmAudioFrame& inputFrame)
    {
        cudaMemcpy(m_buffers.currentInputFrame(), inputFrame.data(), inputFrame.size(), cudaMemcpyHostToDevice);
        processKernel<<<1, 128>>>(m_buffers);
        cudaMemcpy(&m_outputFrame[0], m_buffers.currentOutputFrame(), m_outputFrame.size(), cudaMemcpyDeviceToHost);

        m_buffers.nextFrame();

        return m_outputFrame;
    }
}

#endif
