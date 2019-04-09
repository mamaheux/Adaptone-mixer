#ifndef SIGNAL_PROCESSING_CUDA_CUDA_SOUND_LEVEL_BUFFERS_H
#define SIGNAL_PROCESSING_CUDA_CUDA_SOUND_LEVEL_BUFFERS_H

#include <Utils/ClassMacro.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace adaptone
{
    template<class T>
    class CudaSoundLevelBuffers
    {
        T* m_soundLevels;

        std::size_t m_channelCount;

        bool m_hasOwnership;

    public:
        __host__ explicit CudaSoundLevelBuffers(std::size_t channelCount);
        __host__ CudaSoundLevelBuffers(const CudaSoundLevelBuffers& other);
        __host__ virtual ~CudaSoundLevelBuffers();

        DECLARE_NOT_MOVABLE(CudaSoundLevelBuffers);

        __device__ __host__ T* soundLevels();
        __device__ __host__ std::size_t channelCount();
    };

    template<class T>
    inline __host__ CudaSoundLevelBuffers<T>::CudaSoundLevelBuffers(std::size_t channelCount) :
        m_channelCount(channelCount),
        m_hasOwnership(true)
    {
        cudaMalloc(reinterpret_cast<void**>(&m_soundLevels), m_channelCount);
    }

    template<class T>
    inline __host__ CudaSoundLevelBuffers<T>::CudaSoundLevelBuffers(const CudaSoundLevelBuffers<T>& other) :
        m_soundLevels(other.m_soundLevels),
        m_channelCount(other.m_channelCount),
        m_hasOwnership(false)
    {
    }

    template<class T>
    inline __host__ CudaSoundLevelBuffers<T>::~CudaSoundLevelBuffers()
    {
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
}

#endif
