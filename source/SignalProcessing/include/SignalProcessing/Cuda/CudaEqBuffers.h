#ifndef SIGNAL_PROCESSING_CUDA_CUDA_EQ_BUFFERS_H
#define SIGNAL_PROCESSING_CUDA_CUDA_EQ_BUFFERS_H

#include <SignalProcessing/Filters/BiquadCoefficients.h>

#include <Utils/ClassMacro.h>
#include <Utils/Data/PcmAudioFrame.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace adaptone
{
    template<class T>
    class CudaEqBuffers
    {
        BiquadCoefficients<T>* m_biquadCoefficients;
        T* m_d0;

        std::size_t m_channelCount;
        std::size_t m_filterCountPerChannel;

        bool m_hasOwnership;

    public:
        __host__ CudaEqBuffers(std::size_t channelCount, std::size_t filterCountPerChannel);
        __host__ CudaEqBuffers(const CudaEqBuffers& other);
        __host__ virtual ~CudaEqBuffers();

        DECLARE_NOT_MOVABLE(CudaEqBuffers);

        __device__ __host__ BiquadCoefficients<T>* biquadCoefficients();
        __device__ __host__ BiquadCoefficients<T>* biquadCoefficients(std::size_t channel);
        __device__ __host__ T* d0();

        __device__ __host__ std::size_t channelCount();
        __device__ __host__ std::size_t filterCountPerChannel();
    };

    template<class T>
    inline __host__ CudaEqBuffers<T>::CudaEqBuffers(std::size_t channelCount,
        std::size_t filterCountPerChannel) :
        m_channelCount(channelCount),
        m_filterCountPerChannel(filterCountPerChannel),
        m_hasOwnership(true)
    {
        cudaMalloc(reinterpret_cast<void**>(&m_biquadCoefficients),
            m_channelCount * m_filterCountPerChannel * sizeof(BiquadCoefficients<T>));
        cudaMalloc(reinterpret_cast<void**>(&m_d0), m_channelCount * sizeof(T));
    }

    template<class T>
    inline __host__ CudaEqBuffers<T>::CudaEqBuffers(
        const CudaEqBuffers<T>& other) :
        m_biquadCoefficients(other.m_biquadCoefficients),
        m_d0(other.m_d0),
        m_channelCount(other.m_channelCount),
        m_filterCountPerChannel(other.m_filterCountPerChannel),
        m_hasOwnership(false)
    {
    }

    template<class T>
    inline __host__ CudaEqBuffers<T>::~CudaEqBuffers()
    {
        if (m_hasOwnership)
        {
            cudaFree(m_biquadCoefficients);
            cudaFree(m_d0);
        }
    }

    template<class T>
    inline __device__ __host__ BiquadCoefficients<T>* CudaEqBuffers<T>::biquadCoefficients()
    {
        return m_biquadCoefficients;
    }

    template<class T>
    inline __device__ __host__ BiquadCoefficients<T>* CudaEqBuffers<T>::biquadCoefficients(std::size_t channel)
    {
        return m_biquadCoefficients + channel * m_filterCountPerChannel;
    }

    template<class T>
    inline __device__ __host__ T* CudaEqBuffers<T>::d0()
    {
        return m_d0;
    }

    template<class T>
    inline __device__ __host__ std::size_t CudaEqBuffers<T>::channelCount()
    {
        return m_channelCount;
    }

    template<class T>
    inline __device__ __host__ std::size_t CudaEqBuffers<T>::filterCountPerChannel()
    {
        return m_filterCountPerChannel;
    }
}

#endif
