#ifndef SIGNAL_PROCESSING_CUDA_CUDA_SIGNAL_PROCESSOR_BUFFERS_H
#define SIGNAL_PROCESSING_CUDA_CUDA_SIGNAL_PROCESSOR_BUFFERS_H

#include <SignalProcessing/Cuda/Conversion/PcmToArrayConversion.h>

#include <Utils/ClassMacro.h>
#include <Utils/Data/PcmAudioFrame.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace adaptone
{
    template <class T>
    class CudaSignalProcessorBuffers
    {
        uint8_t* m_inputPcmFrames;
        uint8_t* m_outputPcmFrames;

        T* m_inputFrames;

        T* m_inputGains;
        T* m_mixingGains;
        T* m_outputGains;

        std::size_t m_currentFrameIndex;
        std::size_t m_frameCount;
        std::size_t m_inputPcmFrameSize;
        std::size_t m_outputPcmFrameSize;

        std::size_t m_frameSampleCount;
        std::size_t m_inputChannelCount;
        std::size_t m_outputChannelCount;
        std::size_t m_frameSize;

        std::size_t m_mixingGainsSize;

        PcmToArrayConversionFunctionPointer<T> m_pcmToArrayConversionFunction;

        bool m_hasOwnership;

    public:
        __host__ CudaSignalProcessorBuffers(std::size_t frameCount,
            std::size_t frameSampleCount,
            std::size_t inputChannelCount,
            std::size_t outputChannelCount,
            PcmAudioFrame::Format inputFormat,
            PcmAudioFrame::Format outputFormat);
        __host__ CudaSignalProcessorBuffers(const CudaSignalProcessorBuffers& other);
        __host__ virtual ~CudaSignalProcessorBuffers();

        DECLARE_NOT_MOVABLE(CudaSignalProcessorBuffers);

        __device__ __host__ uint8_t* inputPcmFrames();
        __device__ __host__ uint8_t* currentInputPcmFrame();

        __device__ __host__ uint8_t* outputPcmFrames();
        __device__ __host__ uint8_t* currentOutputPcmFrame();

        __device__ __host__ T* inputFrames();
        __device__ __host__ T* currentInputFrame();

        __device__ __host__ T* inputGains();
        __device__ __host__ T* mixingGains();
        __device__ __host__ T* outputGains();

        __device__ __host__ std::size_t currentFrameIndex();
        __host__ void nextFrame();

        __device__ __host__ std::size_t frameCount();
        __device__ __host__ std::size_t inputPcmFrameSize();
        __device__ __host__ std::size_t outputPcmFrameSize();

        __device__ __host__ std::size_t frameSampleCount();
        __device__ __host__ std::size_t inputChannelCount();
        __device__ __host__ std::size_t outputChannelCount();
        __device__ __host__ std::size_t frameSize();
        __device__ __host__ std::size_t mixingGainsSize();

        __device__ PcmToArrayConversionFunctionPointer<T> pcmToArrayConversionFunction();
    };

    template <class T>
    inline __host__ CudaSignalProcessorBuffers<T>::CudaSignalProcessorBuffers(std::size_t frameCount,
        std::size_t frameSampleCount,
        std::size_t inputChannelCount,
        std::size_t outputChannelCount,
        PcmAudioFrame::Format inputFormat,
        PcmAudioFrame::Format outputFormat) :
        m_currentFrameIndex(0),
        m_inputPcmFrameSize(PcmAudioFrame::size(inputFormat, inputChannelCount, frameSampleCount)),
        m_outputPcmFrameSize(PcmAudioFrame::size(outputFormat, outputChannelCount, frameSampleCount)),
        m_frameCount(frameCount),
        m_frameSampleCount(frameSampleCount),
        m_inputChannelCount(inputChannelCount),
        m_outputChannelCount(outputChannelCount),
        m_frameSize(m_frameSampleCount * m_inputChannelCount),
        m_mixingGainsSize(m_inputChannelCount * m_outputChannelCount),
        m_hasOwnership(true)
    {
        cudaMalloc(reinterpret_cast<void**>(&m_inputPcmFrames), m_inputPcmFrameSize * frameCount);
        cudaMalloc(reinterpret_cast<void**>(&m_outputPcmFrames), m_outputPcmFrameSize * frameCount);

        cudaMalloc(reinterpret_cast<void**>(&m_inputFrames), m_frameSize * frameCount * sizeof(T));

        cudaMalloc(reinterpret_cast<void**>(&m_inputGains), m_inputChannelCount * sizeof(T));
        cudaMalloc(reinterpret_cast<void**>(&m_mixingGains), m_mixingGainsSize * sizeof(T));
        cudaMalloc(reinterpret_cast<void**>(&m_outputGains), m_outputChannelCount * sizeof(T));

        cudaMemset(m_inputGains, 0, m_inputChannelCount * sizeof(T));
        cudaMemset(m_mixingGains, 0, m_mixingGainsSize * sizeof(T));
        cudaMemset(m_outputGains, 0, m_outputChannelCount * sizeof(T));

        m_pcmToArrayConversionFunction = getPcmToArrayConversionFunctionPointer<T>(inputFormat);
    }

    template <class T>
    inline __host__ CudaSignalProcessorBuffers<T>::CudaSignalProcessorBuffers(
        const CudaSignalProcessorBuffers<T>& other) :
        m_inputPcmFrames(other.m_inputPcmFrames),
        m_outputPcmFrames(other.m_outputPcmFrames),
        m_inputFrames(other.m_inputFrames),
        m_inputGains(other.m_inputGains),
        m_mixingGains(other.m_mixingGains),
        m_outputGains(other.m_outputGains),
        m_currentFrameIndex(other.m_currentFrameIndex),
        m_inputPcmFrameSize(other.m_inputPcmFrameSize),
        m_outputPcmFrameSize(other.m_outputPcmFrameSize),
        m_frameCount(other.m_frameCount),
        m_frameSampleCount(other.m_frameSampleCount),
        m_inputChannelCount(other.m_inputChannelCount),
        m_outputChannelCount(other.m_outputChannelCount),
        m_frameSize(other.m_frameSize),
        m_mixingGainsSize(other.m_mixingGainsSize),
        m_pcmToArrayConversionFunction(other.m_pcmToArrayConversionFunction),
        m_hasOwnership(false)
    {
    }

    template <class T>
    inline __host__ CudaSignalProcessorBuffers<T>::~CudaSignalProcessorBuffers()
    {
        if (m_hasOwnership)
        {
            cudaFree(m_inputPcmFrames);
            cudaFree(m_outputPcmFrames);

            cudaFree(m_inputFrames);

            cudaFree(m_inputGains);
            cudaFree(m_mixingGains);
            cudaFree(m_outputGains);
        }
    }

    template <class T>
    inline __device__ __host__ uint8_t* CudaSignalProcessorBuffers<T>::inputPcmFrames()
    {
        return m_inputFrames;
    }

    template <class T>
    inline __device__ __host__ uint8_t* CudaSignalProcessorBuffers<T>::currentInputPcmFrame()
    {
        return m_inputPcmFrames + m_currentFrameIndex * m_inputPcmFrameSize;
    }

    template <class T>
    inline __device__ __host__ uint8_t* CudaSignalProcessorBuffers<T>::outputPcmFrames()
    {
        return m_outputPcmFrames;
    }

    template <class T>
    inline __device__ __host__ uint8_t* CudaSignalProcessorBuffers<T>::currentOutputPcmFrame()
    {
        return m_outputPcmFrames + m_currentFrameIndex * m_outputPcmFrameSize;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::inputFrames()
    {
        return m_inputFrames;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::currentInputFrame()
    {
        return m_inputFrames + m_currentFrameIndex * m_frameSize;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::inputGains()
    {
        return m_inputGains;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::mixingGains()
    {
        return m_mixingGains;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::outputGains()
    {
        return m_outputGains;
    }

    template <class T>
    inline __device__ __host__ std::size_t CudaSignalProcessorBuffers<T>::currentFrameIndex()
    {
        return m_currentFrameIndex;
    }

    template <class T>
    inline __host__ void CudaSignalProcessorBuffers<T>::nextFrame()
    {
        m_currentFrameIndex = (m_currentFrameIndex + 1) % m_frameCount;
    }

    template <class T>
    inline __device__ __host__ std::size_t CudaSignalProcessorBuffers<T>::frameCount()
    {
        return m_frameCount;
    }

    template <class T>
    inline __device__ __host__ std::size_t CudaSignalProcessorBuffers<T>::inputPcmFrameSize()
    {
        return m_inputPcmFrameSize;
    }

    template <class T>
    inline __device__ __host__ std::size_t CudaSignalProcessorBuffers<T>::outputPcmFrameSize()
    {
        return m_outputPcmFrameSize;
    }

    template<class T>
    inline __device__ __host__ std::size_t CudaSignalProcessorBuffers<T>::frameSampleCount()
    {
        return m_frameSampleCount;
    }

    template<class T>
    inline __device__ __host__ std::size_t CudaSignalProcessorBuffers<T>::inputChannelCount()
    {
        return m_inputChannelCount;
    }

    template<class T>
    inline __device__ __host__ std::size_t CudaSignalProcessorBuffers<T>::outputChannelCount()
    {
        return m_outputChannelCount;
    }

    template<class T>
    inline __device__ __host__ std::size_t CudaSignalProcessorBuffers<T>::frameSize()
    {
        return m_frameSize;
    }

    template<class T>
    inline __device__ __host__ std::size_t mixingGainsSize()
    {
        return m_mixingGainsSize;
    }

    template<class T>
    inline __device__ PcmToArrayConversionFunctionPointer<T> CudaSignalProcessorBuffers<T>::pcmToArrayConversionFunction()
    {
        return m_pcmToArrayConversionFunction;
    }
}

#endif
