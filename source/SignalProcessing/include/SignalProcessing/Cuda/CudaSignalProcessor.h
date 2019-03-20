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
        m_buffers(CudaSignalProcessorFrameCount,
            frameSampleCount,
            inputChannelCount,
            outputChannelCount,
            inputFormat,
            outputFormat)
    {
    }

    template<class T>
    CudaSignalProcessor<T>::~CudaSignalProcessor()
    {
    }

    template<class T>
    __global__ void processKernel(CudaSignalProcessorBuffers<T> buffers)
    {
        uint8_t* inputPcmFrame = buffers.currentInputPcmFrame();
        T* inputFrame = buffers.currentInputFrame();
        std::size_t frameSampleCount = buffers.frameSampleCount();
        std::size_t inputChannelCount = buffers.inputChannelCount();

        buffers.pcmToArrayConversionFunction()(inputPcmFrame, inputFrame, frameSampleCount, inputChannelCount);

        //TODO Remove the following code
        int index = threadIdx.x;
        int stride = blockDim.x;

        uint8_t* outputPcmFrame = buffers.currentOutputPcmFrame();

        for (int i = index; i < buffers.inputPcmFrameSize() && i < buffers.outputPcmFrameSize(); i += stride)
        {
            outputPcmFrame[i] = inputPcmFrame[i];
        }
    }

    template<class T>
    const PcmAudioFrame& CudaSignalProcessor<T>::process(const PcmAudioFrame& inputFrame)
    {
        cudaMemcpy(m_buffers.currentInputPcmFrame(), inputFrame.data(), inputFrame.size(), cudaMemcpyHostToDevice);
        processKernel<<<1, 256>>>(m_buffers);
        cudaMemcpy(&m_outputFrame[0], m_buffers.currentOutputPcmFrame(), m_outputFrame.size(), cudaMemcpyDeviceToHost);

        m_buffers.nextFrame();

        return m_outputFrame;
    }
}

#endif
