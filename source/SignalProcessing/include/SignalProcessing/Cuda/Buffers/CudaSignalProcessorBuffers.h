#ifndef SIGNAL_PROCESSING_CUDA_CUDA_SIGNAL_PROCESSOR_BUFFERS_H
#define SIGNAL_PROCESSING_CUDA_CUDA_SIGNAL_PROCESSOR_BUFFERS_H

#include <SignalProcessing/Cuda/Buffers/CudaEqBuffers.h>
#include <SignalProcessing/Cuda/Buffers/CudaSoundLevelBuffers.h>
#include <SignalProcessing/Cuda/Conversion/PcmToArrayConversion.h>
#include <SignalProcessing/Cuda/Conversion/ArrayToPcmConversion.h>
#include <SignalProcessing/SignalProcessorParameters.h>

#include <Utils/ClassMacro.h>
#include <Utils/Data/PcmAudioFrameFormat.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace adaptone
{
    /**
     * Buffer format:
     * c1 = channel 1
     * s1 = sample 1
     * gc1 = gain channel 1
     * gic1oc1 = gain input channel 1 output channel 1
     *
     * m_inputPcmFrames: | frame1 | frame2 | ... |
     *   frame: | c1s1 | c2s1 | c3s1 | ... | c1s2 | c2s2 | c3s2 |... |
     *
     * m_outputPcmFrames: | frame1 | frame2 | ... |
     *   frame: | c1s1 | c2s1 | c3s1 | ... | c1s2 | c2s2 | c3s2 |... |
     *
     * m_inputFrames: | frame1 | frame2 | ... |
     *   frame: | c1s1 | c1s2 | c1s3 | ... | c2s1 | c2s2 | c2s3 | ... |
     *
     * m_inputGainOutputFrames: | frame1 | frame2 | ... |
     *   frame: | c1s1 | c1s2 | c1s3 | ... | c2s1 | c2s2 | c2s3 | ... |
     *
     * m_inputEqOutputFrames: | frame1 | frame2 | ... |
     *   frame: | c1s1 | c1s2 | c1s3 | ... | c2s1 | c2s2 | c2s3 | ... |
     *
     * m_mixingOutputFrames: | frame1 | frame2 | ... |
     *   frame: | c1s1 | c1s2 | c1s3 | ... | c2s1 | c2s2 | c2s3 | ... |
     *
     * m_outputEqOutputFrames: | frame1 | frame2 | ... |
     *   frame: | c1s1 | c1s2 | c1s3 | ... | c2s1 | c2s2 | c2s3 | ... |
     *
     * m_outputFrames: | frame1 | frame2 | ... |
     *   frame: | c1s1 | c1s2 | c3s3 | ... | c2s1 | c2s2 | c3s3 | ... |
     *
     * m_delayedOutputFrames: | frame1 | frame2 | ... |
     *   frame: | c1s1 | c1s2 | c1s3 | ... | c2s1 | c2s2 | c1s3 | ... |
     *
     * m_inputGains: | gc1 | gc2 | gc3 | ... |
     *
     * m_mixingGains: | gic1oc1 | gic2oc1 | gic3oc1 | ... | gic1oc2 | gic2oc2 | gic3oc2 | ... |
     *
     * m_outputGains:| gc1 | gc2 | gc3 | ... |
     */
    template<class T>
    class CudaSignalProcessorBuffers
    {
        uint8_t* m_inputPcmFrames;
        uint8_t* m_outputPcmFrames;

        T* m_inputFrames;
        T* m_inputGainOutputFrames;
        T* m_inputEqOutputFrames;

        T* m_mixingOutputFrames;
        T* m_outputEqOutputFrames;
        T* m_uniformizationEqOutputFrames;
        T* m_outputFrames;
        T* m_delayedOutputFrames;

        T* m_inputGains;
        CudaEqBuffers<T> m_inputEqBuffers;
        T* m_mixingGains;
        CudaEqBuffers<T> m_outputEqBuffers;
        CudaEqBuffers<T> m_uniformizationEqBuffers;
        T* m_outputGains;
        std::size_t* m_outputDelays;

        CudaSoundLevelBuffers<T> m_inputGainSoundLevelBuffers;
        CudaSoundLevelBuffers<T> m_inputEqSoundLevelBuffers;
        CudaSoundLevelBuffers<T> m_outputGainSoundLevelBuffers;

        std::size_t m_currentFrameIndex;
        std::size_t m_currentDelayedOutputFrameIndex;
        std::size_t m_frameCount;
        std::size_t m_inputPcmFrameSize;
        std::size_t m_outputPcmFrameSize;

        std::size_t m_frameSampleCount;
        std::size_t m_inputChannelCount;
        std::size_t m_outputChannelCount;
        std::size_t m_inputFrameSize;
        std::size_t m_outputFrameSize;

        std::size_t m_mixingGainsSize;
        std::size_t m_delayedOutputFrameCount;
        std::size_t m_maxOutputDelay;

        PcmAudioFrameFormat m_inputFormat;
        PcmAudioFrameFormat m_outputFormat;

        bool m_hasOwnership;

    public:
        __host__ CudaSignalProcessorBuffers(const SignalProcessorParameters& parameters,
            std::size_t frameCount,
            std::size_t eqFilterCountPerChannel);
        __host__ CudaSignalProcessorBuffers(const CudaSignalProcessorBuffers& other);
        __host__ virtual ~CudaSignalProcessorBuffers();

        DECLARE_NOT_MOVABLE(CudaSignalProcessorBuffers);

        __device__ __host__ uint8_t* inputPcmFrames();
        __device__ __host__ uint8_t* currentInputPcmFrame();

        __device__ __host__ uint8_t* outputPcmFrames();
        __device__ __host__ uint8_t* currentOutputPcmFrame();

        __device__ __host__ T* inputFrames();
        __device__ __host__ T* currentInputFrame();

        __device__ __host__ T* inputGainOutputFrames();
        __device__ __host__ T* currentInputGainOutputFrame();

        __device__ __host__ T* inputEqOutputFrames();
        __device__ __host__ T* currentInputEqOutputFrame();

        __device__ __host__ T* mixingOutputFrames();
        __device__ __host__ T* currentMixingOutputFrame();

        __device__ __host__ T* outputEqOutputFrames();
        __device__ __host__ T* currentOutputEqOutputFrame();
        __device__ __host__ T* uniformizationEqOutputFrames();
        __device__ __host__ T* currentUniformizationEqOutputFrame();

        __device__ __host__ T* outputFrames();
        __device__ __host__ T* currentOutputFrame();

        __device__ __host__ T* delayedOutputFrames();
        __device__ __host__ T* currentDelayedOutputFrame();

        __device__ __host__ T* inputGains();
        __device__ __host__ CudaEqBuffers<T>& inputEqBuffers();
        __device__ __host__ T* mixingGains();
        __device__ __host__ CudaEqBuffers<T>& outputEqBuffers();
        __device__ __host__ CudaEqBuffers<T>& uniformizationEqBuffers();
        __device__ __host__ T* outputGains();
        __device__ __host__ std::size_t* outputDelays();

        __device__ __host__ CudaSoundLevelBuffers<T>& inputGainSoundLevelBuffers();
        __device__ __host__ CudaSoundLevelBuffers<T>& inputEqSoundLevelBuffers();
        __device__ __host__ CudaSoundLevelBuffers<T>& outputGainSoundLevelBuffers();

        __device__ __host__ std::size_t currentFrameIndex();
        __device__ __host__ std::size_t currentDelayedOutputFrameIndex();
        __host__ void nextFrame();

        __device__ __host__ std::size_t frameCount();
        __device__ __host__ std::size_t inputPcmFrameSize();
        __device__ __host__ std::size_t outputPcmFrameSize();

        __device__ __host__ std::size_t frameSampleCount();
        __device__ __host__ std::size_t inputChannelCount();
        __device__ __host__ std::size_t outputChannelCount();
        __device__ __host__ std::size_t inputFrameSize();
        __device__ __host__ std::size_t outputFrameSize();

        __device__ __host__ std::size_t mixingGainsSize();
        __device__ __host__ std::size_t delayedOutputFrameCount();

        __device__ __host__ PcmAudioFrameFormat inputFormat();
        __device__ __host__ PcmAudioFrameFormat outputFormat();

        __host__ void updateInputGains(const T* data);
        __host__ void updateMixingGains(const T* data);
        __host__ void updateOutputGains(const T* data);
        __host__ void updateOutputDelays(const std::size_t* data);

        __host__ void copyInputFrame(const PcmAudioFrame& frame);
        __host__ void copyOutputFrame(PcmAudioFrame& frame);
        __host__ void copyCurrentInputEqOutputFrame(T* data);
    };

    template<class T>
    inline __host__ CudaSignalProcessorBuffers<T>::CudaSignalProcessorBuffers(
        const SignalProcessorParameters& parameters,
        std::size_t frameCount,
        std::size_t eqFilterCountPerChannel) :
        m_currentFrameIndex(0),
        m_currentDelayedOutputFrameIndex(0),
        m_inputPcmFrameSize(size(parameters.inputFormat(), parameters.inputChannelCount(),
            parameters.frameSampleCount())),
        m_outputPcmFrameSize(size(parameters.outputFormat(), parameters.outputChannelCount(),
            parameters.frameSampleCount())),

        m_frameCount(frameCount),
        m_frameSampleCount(parameters.frameSampleCount()),
        m_inputChannelCount(parameters.inputChannelCount()),
        m_outputChannelCount(parameters.outputChannelCount()),
        m_inputFrameSize(m_frameSampleCount * m_inputChannelCount),
        m_outputFrameSize(m_frameSampleCount * m_outputChannelCount),
        m_mixingGainsSize(m_inputChannelCount * m_outputChannelCount),
        m_delayedOutputFrameCount(parameters.maxOutputDelay() / m_frameSampleCount + 2),
        m_maxOutputDelay(parameters.maxOutputDelay()),

        m_inputEqBuffers(parameters.inputChannelCount(), eqFilterCountPerChannel, frameCount,
            parameters.frameSampleCount()),
        m_outputEqBuffers(parameters.outputChannelCount(), eqFilterCountPerChannel, frameCount,
            parameters.frameSampleCount()),
        m_uniformizationEqBuffers(parameters.outputChannelCount(), eqFilterCountPerChannel, frameCount,
            parameters.frameSampleCount()),

        m_inputGainSoundLevelBuffers(parameters.inputChannelCount(), parameters.frameSampleCount()),
        m_inputEqSoundLevelBuffers(parameters.inputChannelCount(), parameters.frameSampleCount()),
        m_outputGainSoundLevelBuffers(parameters.outputChannelCount(), parameters.frameSampleCount()),

        m_inputFormat(parameters.inputFormat()),
        m_outputFormat(parameters.outputFormat()),

        m_hasOwnership(true)
    {
        cudaMalloc(reinterpret_cast<void**>(&m_inputPcmFrames), m_inputPcmFrameSize * frameCount);
        cudaMalloc(reinterpret_cast<void**>(&m_outputPcmFrames), m_outputPcmFrameSize * frameCount);

        cudaMalloc(reinterpret_cast<void**>(&m_inputFrames), m_inputFrameSize * frameCount * sizeof(T));
        cudaMalloc(reinterpret_cast<void**>(&m_inputGainOutputFrames), m_inputFrameSize * frameCount * sizeof(T));
        cudaMalloc(reinterpret_cast<void**>(&m_inputEqOutputFrames), m_inputFrameSize * frameCount * sizeof(T));

        cudaMalloc(reinterpret_cast<void**>(&m_mixingOutputFrames), m_outputFrameSize * frameCount * sizeof(T));
        cudaMalloc(reinterpret_cast<void**>(&m_outputEqOutputFrames), m_outputFrameSize * frameCount * sizeof(T));
        cudaMalloc(reinterpret_cast<void**>(&m_uniformizationEqOutputFrames),
            m_outputFrameSize * frameCount * sizeof(T));
        cudaMalloc(reinterpret_cast<void**>(&m_outputFrames), m_outputFrameSize * frameCount * sizeof(T));
        cudaMalloc(reinterpret_cast<void**>(&m_delayedOutputFrames),
            m_outputFrameSize * m_delayedOutputFrameCount * sizeof(T));

        cudaMalloc(reinterpret_cast<void**>(&m_inputGains), m_inputChannelCount * sizeof(T));
        cudaMalloc(reinterpret_cast<void**>(&m_mixingGains), m_mixingGainsSize * sizeof(T));
        cudaMalloc(reinterpret_cast<void**>(&m_outputGains), m_outputChannelCount * sizeof(T));
        cudaMalloc(reinterpret_cast<void**>(&m_outputDelays), m_outputChannelCount * sizeof(std::size_t));

        cudaMemset(m_inputGains, 0, m_inputChannelCount * sizeof(T));
        cudaMemset(m_mixingGains, 0, m_mixingGainsSize * sizeof(T));
        cudaMemset(m_outputGains, 0, m_outputChannelCount * sizeof(T));
        cudaMemset(m_outputDelays, 0, m_outputChannelCount * sizeof(std::size_t));
    }

    template<class T>
    inline __host__ CudaSignalProcessorBuffers<T>::CudaSignalProcessorBuffers(
        const CudaSignalProcessorBuffers<T>& other) :
        m_inputPcmFrames(other.m_inputPcmFrames),
        m_outputPcmFrames(other.m_outputPcmFrames),

        m_inputFrames(other.m_inputFrames),
        m_inputGainOutputFrames(other.m_inputGainOutputFrames),
        m_inputEqOutputFrames(other.m_inputEqOutputFrames),

        m_mixingOutputFrames(other.m_mixingOutputFrames),
        m_outputEqOutputFrames(other.m_outputEqOutputFrames),
        m_uniformizationEqOutputFrames(other.m_uniformizationEqOutputFrames),
        m_outputFrames(other.m_outputFrames),
        m_delayedOutputFrames(other.m_delayedOutputFrames),

        m_inputGains(other.m_inputGains),
        m_inputEqBuffers(other.m_inputEqBuffers),
        m_mixingGains(other.m_mixingGains),
        m_outputEqBuffers(other.m_outputEqBuffers),
        m_uniformizationEqBuffers(other.m_uniformizationEqBuffers),
        m_outputGains(other.m_outputGains),
        m_outputDelays(other.m_outputDelays),
        m_inputGainSoundLevelBuffers(other.m_inputGainSoundLevelBuffers),
        m_inputEqSoundLevelBuffers(other.m_inputEqSoundLevelBuffers),
        m_outputGainSoundLevelBuffers(other.m_outputGainSoundLevelBuffers),

        m_currentFrameIndex(other.m_currentFrameIndex),
        m_currentDelayedOutputFrameIndex(other.m_currentDelayedOutputFrameIndex),
        m_inputPcmFrameSize(other.m_inputPcmFrameSize),
        m_outputPcmFrameSize(other.m_outputPcmFrameSize),
        m_frameCount(other.m_frameCount),
        m_frameSampleCount(other.m_frameSampleCount),
        m_inputChannelCount(other.m_inputChannelCount),
        m_outputChannelCount(other.m_outputChannelCount),
        m_inputFrameSize(other.m_inputFrameSize),
        m_outputFrameSize(other.m_outputFrameSize),
        m_mixingGainsSize(other.m_mixingGainsSize),
        m_delayedOutputFrameCount(other.m_delayedOutputFrameCount),
        m_maxOutputDelay(other.m_maxOutputDelay),

        m_inputFormat(other.m_inputFormat),
        m_outputFormat(other.m_outputFormat),

        m_hasOwnership(false)
    {
    }

    template<class T>
    inline __host__ CudaSignalProcessorBuffers<T>::~CudaSignalProcessorBuffers()
    {
        if (m_hasOwnership)
        {
            cudaFree(m_inputPcmFrames);
            cudaFree(m_outputPcmFrames);

            cudaFree(m_inputFrames);
            cudaFree(m_inputGainOutputFrames);
            cudaFree(m_inputEqOutputFrames);

            cudaFree(m_mixingOutputFrames);
            cudaFree(m_outputEqOutputFrames);
            cudaFree(m_uniformizationEqOutputFrames);
            cudaFree(m_outputFrames);
            cudaFree(m_delayedOutputFrames);

            cudaFree(m_inputGains);
            cudaFree(m_mixingGains);
            cudaFree(m_outputGains);
            cudaFree(m_outputDelays);
        }
    }

    template<class T>
    inline __device__ __host__ uint8_t* CudaSignalProcessorBuffers<T>::inputPcmFrames()
    {
        return m_inputFrames;
    }

    template<class T>
    inline __device__ __host__ uint8_t* CudaSignalProcessorBuffers<T>::currentInputPcmFrame()
    {
        return m_inputPcmFrames + m_currentFrameIndex * m_inputPcmFrameSize;
    }

    template<class T>
    inline __device__ __host__ uint8_t* CudaSignalProcessorBuffers<T>::outputPcmFrames()
    {
        return m_outputPcmFrames;
    }

    template<class T>
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
        return m_inputFrames + m_currentFrameIndex * m_inputFrameSize;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::inputGainOutputFrames()
    {
        return m_inputGainOutputFrames;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::currentInputGainOutputFrame()
    {
        return m_inputGainOutputFrames + m_currentFrameIndex * m_inputFrameSize;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::inputEqOutputFrames()
    {
        return m_inputEqOutputFrames;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::currentInputEqOutputFrame()
    {
        return m_inputEqOutputFrames + m_currentFrameIndex * m_inputFrameSize;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::mixingOutputFrames()
    {
        return m_mixingOutputFrames;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::currentMixingOutputFrame()
    {
        return m_mixingOutputFrames + m_currentFrameIndex * m_outputFrameSize;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::outputEqOutputFrames()
    {
        return m_outputEqOutputFrames;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::currentOutputEqOutputFrame()
    {
        return m_outputEqOutputFrames + m_currentFrameIndex * m_outputFrameSize;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::uniformizationEqOutputFrames()
    {
        return m_uniformizationEqOutputFrames;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::currentUniformizationEqOutputFrame()
    {
        return m_uniformizationEqOutputFrames + m_currentFrameIndex * m_outputFrameSize;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::currentOutputFrame()
    {
        return m_outputFrames + m_currentFrameIndex * m_inputFrameSize;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::outputFrames()
    {
        return m_outputFrames;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::delayedOutputFrames()
    {
        return m_delayedOutputFrames;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::currentDelayedOutputFrame()
    {
        return m_delayedOutputFrames + m_currentDelayedOutputFrameIndex * m_outputFrameSize;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::inputGains()
    {
        return m_inputGains;
    }

    template<class T>
    inline __device__ __host__ CudaEqBuffers<T>& CudaSignalProcessorBuffers<T>::inputEqBuffers()
    {
        return m_inputEqBuffers;
    }

    template<class T>
    inline __device__ __host__ CudaSoundLevelBuffers<T>& CudaSignalProcessorBuffers<T>::inputGainSoundLevelBuffers()
    {
        return m_inputGainSoundLevelBuffers;
    }

    template<class T>
    inline __device__ __host__ CudaSoundLevelBuffers<T>& CudaSignalProcessorBuffers<T>::inputEqSoundLevelBuffers()
    {
        return m_inputEqSoundLevelBuffers;
    }

    template<class T>
    inline __device__ __host__ CudaSoundLevelBuffers<T>& CudaSignalProcessorBuffers<T>::outputGainSoundLevelBuffers()
    {
        return m_outputGainSoundLevelBuffers;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::mixingGains()
    {
        return m_mixingGains;
    }

    template<class T>
    inline __device__ __host__ CudaEqBuffers<T>& CudaSignalProcessorBuffers<T>::outputEqBuffers()
    {
        return m_outputEqBuffers;
    }

    template<class T>
    inline __device__ __host__ CudaEqBuffers<T>& CudaSignalProcessorBuffers<T>::uniformizationEqBuffers()
    {
        return m_uniformizationEqBuffers;
    }

    template<class T>
    inline __device__ __host__ T* CudaSignalProcessorBuffers<T>::outputGains()
    {
        return m_outputGains;
    }

    template<class T>
    inline __device__ __host__ std::size_t* CudaSignalProcessorBuffers<T>::outputDelays()
    {
        return m_outputDelays;
    }

    template<class T>
    inline __device__ __host__ std::size_t CudaSignalProcessorBuffers<T>::currentFrameIndex()
    {
        return m_currentFrameIndex;
    }

    template<class T>
    inline __device__ __host__ std::size_t CudaSignalProcessorBuffers<T>::currentDelayedOutputFrameIndex()
    {
        return m_currentDelayedOutputFrameIndex;
    }

    template<class T>
    inline __host__ void CudaSignalProcessorBuffers<T>::nextFrame()
    {
        m_currentFrameIndex = (m_currentFrameIndex + 1) % m_frameCount;
        m_currentDelayedOutputFrameIndex = (m_currentDelayedOutputFrameIndex + 1) % m_delayedOutputFrameCount;
    }

    template<class T>
    inline __device__ __host__ std::size_t CudaSignalProcessorBuffers<T>::frameCount()
    {
        return m_frameCount;
    }

    template<class T>
    inline __device__ __host__ std::size_t CudaSignalProcessorBuffers<T>::inputPcmFrameSize()
    {
        return m_inputPcmFrameSize;
    }

    template<class T>
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
    inline __device__ __host__ std::size_t CudaSignalProcessorBuffers<T>::inputFrameSize()
    {
        return m_inputFrameSize;
    }

    template<class T>
    inline __device__ __host__ std::size_t CudaSignalProcessorBuffers<T>::outputFrameSize()
    {
        return m_outputFrameSize;
    }

    template<class T>
    inline __device__ __host__ std::size_t CudaSignalProcessorBuffers<T>::mixingGainsSize()
    {
        return m_mixingGainsSize;
    }

    template<class T>
    inline __device__ __host__ std::size_t CudaSignalProcessorBuffers<T>::delayedOutputFrameCount()
    {
        return m_delayedOutputFrameCount;
    }

    template<class T>
    inline __device__ __host__ PcmAudioFrameFormat CudaSignalProcessorBuffers<T>::inputFormat()
    {
        return m_inputFormat;
    }

    template<class T>
    inline __device__ __host__ PcmAudioFrameFormat CudaSignalProcessorBuffers<T>::outputFormat()
    {
        return m_outputFormat;
    }

    template<class T>
    inline __host__ void CudaSignalProcessorBuffers<T>::updateInputGains(const T* data)
    {
        cudaMemcpy(m_inputGains, data, m_inputChannelCount * sizeof(T), cudaMemcpyHostToDevice);
    }

    template<class T>
    inline __host__ void CudaSignalProcessorBuffers<T>::updateMixingGains(const T* data)
    {
        cudaMemcpy(m_mixingGains, data, m_mixingGainsSize * sizeof(T), cudaMemcpyHostToDevice);
    }

    template<class T>
    inline __host__ void CudaSignalProcessorBuffers<T>::updateOutputGains(const T* data)
    {
        cudaMemcpy(m_outputGains, data, m_outputChannelCount * sizeof(T), cudaMemcpyHostToDevice);
    }

    template<class T>
    inline __host__ void CudaSignalProcessorBuffers<T>::updateOutputDelays(const std::size_t* data)
    {
        cudaMemcpy(m_outputDelays, data, m_outputChannelCount * sizeof(T), cudaMemcpyHostToDevice);
    }

    template<class T>
    inline __host__ void CudaSignalProcessorBuffers<T>::copyInputFrame(const PcmAudioFrame& frame)
    {
        cudaMemcpy(currentInputPcmFrame(), frame.data(), frame.size(), cudaMemcpyHostToDevice);
    }

    template<class T>
    inline __host__ void CudaSignalProcessorBuffers<T>::copyOutputFrame(PcmAudioFrame& frame)
    {
        cudaMemcpy(frame.data(), currentOutputPcmFrame(), frame.size(), cudaMemcpyDeviceToHost);
    }

    template<class T>
    inline __host__ void CudaSignalProcessorBuffers<T>::copyCurrentInputEqOutputFrame(T* data)
    {
        cudaMemcpy(data, currentInputEqOutputFrame(), m_inputFrameSize * sizeof(T), cudaMemcpyDeviceToHost);
    }
}

#endif
