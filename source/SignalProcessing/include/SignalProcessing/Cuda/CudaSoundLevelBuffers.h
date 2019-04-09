#ifndef SIGNAL_PROCESSING_CUDA_CUDA_SOUND_LEVEL_BUFFERS_H
#define SIGNAL_PROCESSING_CUDA_CUDA_SOUND_LEVEL_BUFFERS_H

#include <Utils/ClassMacro.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace adaptone
{
    /**
     * Buffer format:
     * c1mv = channel 1 maximum value
     *
     * m_soundLevels: | c1mv | c2mv | c3mv | ... | c16mv |
     *
     */
    template<class T>
    class CudaSoundLevelBuffers
    {
        T* m_soundLevels;

        std::size_t m_channelCount;
        std::size_t m_frameSampleCount;

        bool m_hasOwnership;

    public:
        __host__ CudaSoundLevelBuffers(std::size_t channelCount, std::size_t frameSampleCount);
        __host__ CudaSoundLevelBuffers(const CudaSoundLevelBuffers& other);
        __host__ virtual ~CudaSoundLevelBuffers();

        DECLARE_NOT_MOVABLE(CudaSoundLevelBuffers);

        __device__ __host__ T* soundLevels();
        __device__ __host__ std::size_t channelCount();
        __device__ __host__ std::size_t frameSampleCount();

        __host__ void toVector(std::vector<T>& soundLevels);
        __host__ void resetBuffer();
    };

    template<class T>
    inline __host__ CudaSoundLevelBuffers<T>::CudaSoundLevelBuffers(std::size_t channelCount, std::size_t frameSampleCount) :
        m_channelCount(channelCount),
        m_frameSampleCount(frameSampleCount),
        m_hasOwnership(true)
    {
        cudaMalloc(reinterpret_cast<void**>(&m_soundLevels), m_channelCount * sizeof(T));
    }

    template<class T>
    inline __host__ CudaSoundLevelBuffers<T>::CudaSoundLevelBuffers(const CudaSoundLevelBuffers<T>& other) :
        m_soundLevels(other.m_soundLevels),
        m_channelCount(other.m_channelCount),
        m_frameSampleCount(other.m_frameSampleCount),
        m_hasOwnership(false)
    {
    }

    template<class T>
    inline __host__ CudaSoundLevelBuffers<T>::~CudaSoundLevelBuffers()
    {
        cudaFree(m_soundLevels);
    }

    template<class T>
    inline __device__ __host__ T* CudaSoundLevelBuffers<T>::soundLevels()
    {
        return m_soundLevels;
    }

    template<class T>
    inline __device__ __host__ std::size_t CudaSoundLevelBuffers<T>::channelCount()
    {
        return m_channelCount;
    }

    template<class T>
    inline __device__ __host__ std::size_t CudaSoundLevelBuffers<T>::frameSampleCount()
    {
        return m_frameSampleCount;
    }

    template<class T>
    inline __host__ void CudaSoundLevelBuffers<T>::toVector(std::vector<T>& soundLevels)
    {
        cudaMemcpy(soundLevels.data(),
            m_soundLevels,
            m_channelCount * sizeof(T),
            cudaMemcpyDeviceToHost);
    }

    template<class T>
    inline __host__ void CudaSoundLevelBuffers<T>::resetBuffer()
    {
        for (std::size_t i = 0; i < m_channelCount; i++)
        {
            m_soundLevels[i] = 0;
        }
    }
}

#endif
