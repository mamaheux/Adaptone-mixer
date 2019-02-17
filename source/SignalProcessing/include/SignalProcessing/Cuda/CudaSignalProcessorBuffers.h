#ifndef SIGNAL_PROCESSING_CUDA_CUDA_SIGNAL_PROCESSOR_BUFFERS_H
#define SIGNAL_PROCESSING_CUDA_CUDA_SIGNAL_PROCESSOR_BUFFERS_H

#include <Utils/ClassMacro.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace adaptone
{
    template <class T>
    class CudaSignalProcessorBuffers
    {
        uint8_t* m_inputFrames;
        uint8_t* m_outputFrames;

        size_t m_currentFrameIndex;
        std::size_t m_frameCount;
        std::size_t m_inputFrameSize;
        std::size_t m_outputFrameSize;

        bool m_hasOwnership;

    public:
        __host__ CudaSignalProcessorBuffers(std::size_t inputFrameSize,
            std::size_t outputFrameSize,
            std::size_t frameCount);
        __host__ CudaSignalProcessorBuffers(const CudaSignalProcessorBuffers& other);
        __host__ virtual ~CudaSignalProcessorBuffers();

        DECLARE_NOT_MOVABLE(CudaSignalProcessorBuffers);

        __device__ __host__ uint8_t* inputFrames();
        __device__ __host__ uint8_t* currentInputFrame();

        __device__ __host__ uint8_t* outputFrames();
        __device__ __host__ uint8_t* currentOutputFrame();

        __device__ __host__ std::size_t currentFrameIndex();
        __host__ void nextFrame();

        __device__ __host__ std::size_t frameCount();
        __device__ __host__ std::size_t inputFrameSize();
        __device__ __host__ std::size_t outputFrameSize();
    };

    template <class T>
    inline __host__ CudaSignalProcessorBuffers<T>::CudaSignalProcessorBuffers(std::size_t inputFrameSize,
        std::size_t outputFrameSize,
        std::size_t frameCount) :
        m_currentFrameIndex(0),
        m_inputFrameSize(inputFrameSize),
        m_outputFrameSize(outputFrameSize),
        m_frameCount(frameCount),
        m_hasOwnership(true)
    {
        cudaMalloc(reinterpret_cast<void**>(&m_inputFrames), inputFrameSize * frameCount);
        cudaMalloc(reinterpret_cast<void**>(&m_outputFrames), outputFrameSize * frameCount);
    }

    template <class T>
    inline __host__ CudaSignalProcessorBuffers<T>::CudaSignalProcessorBuffers(
        const CudaSignalProcessorBuffers<T>& other) :
        m_inputFrames(other.m_inputFrames),
        m_outputFrames(other.m_outputFrames),
        m_currentFrameIndex(other.m_currentFrameIndex),
        m_inputFrameSize(other.m_inputFrameSize),
        m_outputFrameSize(other.m_outputFrameSize),
        m_frameCount(other.m_frameCount),
        m_hasOwnership(false)
    {
    }

    template <class T>
    __host__ CudaSignalProcessorBuffers<T>::~CudaSignalProcessorBuffers()
    {
        if (m_hasOwnership)
        {
            cudaFree(m_inputFrames);
            cudaFree(m_outputFrames);
        }
    }

    template <class T>
    inline __device__ __host__ uint8_t* CudaSignalProcessorBuffers<T>::inputFrames()
    {
        return m_inputFrames;
    }

    template <class T>
    inline __device__ __host__ uint8_t* CudaSignalProcessorBuffers<T>::currentInputFrame()
    {
        return m_inputFrames + m_currentFrameIndex * m_inputFrameSize;
    }

    template <class T>
    inline __device__ __host__ uint8_t* CudaSignalProcessorBuffers<T>::outputFrames()
    {
        return m_outputFrames;
    }

    template <class T>
    inline __device__ __host__ uint8_t* CudaSignalProcessorBuffers<T>::currentOutputFrame()
    {
        return m_outputFrames + m_currentFrameIndex * m_outputFrameSize;
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
    inline __device__ __host__ std::size_t CudaSignalProcessorBuffers<T>::inputFrameSize()
    {
        return m_inputFrameSize;
    }

    template <class T>
    inline __device__ __host__ std::size_t CudaSignalProcessorBuffers<T>::outputFrameSize()
    {
        return m_outputFrameSize;
    }
}

#endif
