#ifndef SIGNAL_PROCESSING_CUDA_CUDA_SIGNAL_PROCESSOR_H
#define SIGNAL_PROCESSING_CUDA_CUDA_SIGNAL_PROCESSOR_H

#include <SignalProcessing/ProcessingDataType.h>
#include <SignalProcessing/SpecificSignalProcessor.h>
#include <SignalProcessing/Cuda/CudaSignalProcessorBuffers.h>
#include <SignalProcessing/Cuda/Processing/EqProcessing.h>
#include <SignalProcessing/Cuda/Processing/SoundLevelProcessing.h>
#include <SignalProcessing/Parameters/EqParameters.h>
#include <SignalProcessing/Parameters/GainParameters.h>
#include <SignalProcessing/Parameters/MixingParameters.h>
#include <SignalProcessing/AnalysisDispatcher.h>

#include <Utils/ClassMacro.h>
#include <Utils/Data/PcmAudioFrame.h>
#include <Utils/Functional/FunctionQueue.h>

#include <cuda_runtime.h>

#include <memory>

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

        GainParameters<T> m_inputGainParameters;
        EqParameters<T> m_inputEqParameters;
        MixingParameters<T> m_mixingGainParameters;
        EqParameters<T> m_outputEqParameters;
        GainParameters<T> m_outputGainParameters;

        FunctionQueue<bool()> m_updateFunctionQueue;

        std::size_t m_frameSampleCounter;
        std::size_t m_soundLevelLength;
        std::map<AnalysisDispatcher::SoundLevelType, std::vector<T>> m_soundLevels;
        std::shared_ptr<AnalysisDispatcher> m_analysisDispatcher;

    public:
        CudaSignalProcessor(std::size_t frameSampleCount,
            std::size_t sampleFrequency,
            std::size_t inputChannelCount,
            std::size_t outputChannelCount,
            PcmAudioFrame::Format inputFormat,
            PcmAudioFrame::Format outputFormat,
            std::size_t parametricEqFilterCount,
            const std::vector<double>& eqCenterFrequencies,
            std::size_t soundLevelLength,
            std::shared_ptr<AnalysisDispatcher> analysisDispatcher);
        ~CudaSignalProcessor() override;

        DECLARE_NOT_COPYABLE(CudaSignalProcessor);
        DECLARE_NOT_MOVABLE(CudaSignalProcessor);

        void setInputGain(std::size_t channel, double gainDb) override;
        void setInputGains(const std::vector<double>& gainsDb) override;

        void setInputParametricEqParameters(std::size_t channel,
            const std::vector<ParametricEqParameters>& parameters) override;
        void setInputGraphicEqGains(std::size_t channel, const std::vector<double>& gainsDb) override;

        void setMixingGain(std::size_t inputChannel, std::size_t outputChannel, double gainDb) override;
        void setMixingGains(std::size_t outputChannel, const std::vector<double>& gainsDb) override;
        void setMixingGains(const std::vector<double>& gainsDb) override;

        void setOutputParametricEqParameters(std::size_t channel,
            const std::vector<ParametricEqParameters>& parameters) override;
        void setOutputGraphicEqGains(std::size_t channel, const std::vector<double>& gainsDb) override;

        void setOutputGain(std::size_t channel, double gainDb) override;
        void setOutputGains(const std::vector<double>& gainsDb) override;

        const PcmAudioFrame& process(const PcmAudioFrame& inputFrame) override;

    private:
        void pushInputGainUpdate();
        void pushInputEqUpdate(std::size_t channel);
        void pushMixingGainUpdate();
        void pushOutputEqUpdate(std::size_t channel);
        void pushOutputGainUpdate();

        void notifySoundLevelUpdateIfNeeded();
    };

    template<class T>
    CudaSignalProcessor<T>::CudaSignalProcessor(size_t frameSampleCount,
        size_t sampleFrequency,
        size_t inputChannelCount,
        size_t outputChannelCount,
        PcmAudioFrame::Format inputFormat,
        PcmAudioFrame::Format outputFormat,
        std::size_t parametricEqFilterCount,
        const std::vector<double>& eqCenterFrequencies,
        std::size_t soundLevelLength,
        std::shared_ptr<AnalysisDispatcher> analysisDispatcher) :
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
            outputFormat,
            2 * eqCenterFrequencies.size()),
        m_inputGainParameters(inputChannelCount),
        m_inputEqParameters(sampleFrequency, parametricEqFilterCount, eqCenterFrequencies, inputChannelCount),
        m_mixingGainParameters(inputChannelCount, outputChannelCount),
        m_outputEqParameters(sampleFrequency, parametricEqFilterCount, eqCenterFrequencies, outputChannelCount),
        m_outputGainParameters(outputChannelCount),

        m_frameSampleCounter(0),
        m_soundLevelLength(soundLevelLength),
        m_analysisDispatcher(analysisDispatcher)
    {
        m_soundLevels[AnalysisDispatcher::SoundLevelType::InputGain] = std::vector<T>(inputChannelCount);
        m_soundLevels[AnalysisDispatcher::SoundLevelType::InputEq] = std::vector<T>(inputChannelCount);
        m_soundLevels[AnalysisDispatcher::SoundLevelType::OutputGain] = std::vector<T>(outputChannelCount);

        pushInputGainUpdate();
        pushMixingGainUpdate();
        pushOutputGainUpdate();

        for (std::size_t i = 0; i < inputChannelCount; i++)
        {
            pushInputEqUpdate(i);
        }

        for (std::size_t i = 0; i < outputChannelCount; i++)
        {
            pushOutputEqUpdate(i);
        }

        while (m_updateFunctionQueue.size() > 0)
        {
            m_updateFunctionQueue.execute();
        }
    }

    template<class T>
    CudaSignalProcessor<T>::~CudaSignalProcessor()
    {
    }

    template<class T>
    void CudaSignalProcessor<T>::setInputGain(std::size_t channel, double gainDb)
    {
        m_inputGainParameters.setGain(channel, static_cast<T>(gainDb));
        pushInputGainUpdate();
    }

    template<class T>
    void CudaSignalProcessor<T>::setInputGains(const std::vector<double>& gainsDb)
    {
        m_inputGainParameters.setGains(std::vector<T>(gainsDb.begin(), gainsDb.end()));
        pushInputGainUpdate();
    }

    template<class T>
    void CudaSignalProcessor<T>::setInputParametricEqParameters(std::size_t channel,
        const std::vector<ParametricEqParameters>& parameters)
    {
        m_inputEqParameters.setParametricEqParameters(channel, parameters);
        pushInputEqUpdate(channel);
    }

    template<class T>
    void CudaSignalProcessor<T>::setInputGraphicEqGains(std::size_t channel, const std::vector<double>& gainsDb)
    {
        m_inputEqParameters.setGraphicEqGains(channel, gainsDb);
        pushInputEqUpdate(channel);
    }

    template<class T>
    void CudaSignalProcessor<T>::setMixingGain(std::size_t inputChannel, std::size_t outputChannel, double gainDb)
    {
        m_mixingGainParameters.setGain(inputChannel, outputChannel, static_cast<T>(gainDb));
        pushMixingGainUpdate();
    }

    template<class T>
    void CudaSignalProcessor<T>::setMixingGains(std::size_t outputChannel, const std::vector<double>& gainsDb)
    {
        m_mixingGainParameters.setGains(outputChannel, std::vector<T>(gainsDb.begin(), gainsDb.end()));
        pushMixingGainUpdate();
    }

    template<class T>
    void CudaSignalProcessor<T>::setMixingGains(const std::vector<double>& gainsDb)
    {
        m_mixingGainParameters.setGains(std::vector<T>(gainsDb.begin(), gainsDb.end()));
        pushMixingGainUpdate();
    }

    template<class T>
    void CudaSignalProcessor<T>::setOutputParametricEqParameters(std::size_t channel,
        const std::vector<ParametricEqParameters>& parameters)
    {
        m_outputEqParameters.setParametricEqParameters(channel, parameters);
        pushOutputEqUpdate(channel);
    }

    template<class T>
    void CudaSignalProcessor<T>::setOutputGraphicEqGains(std::size_t channel, const std::vector<double>& gainsDb)
    {
        m_outputEqParameters.setGraphicEqGains(channel, gainsDb);
        pushOutputEqUpdate(channel);
    }

    template<class T>
    void CudaSignalProcessor<T>::setOutputGain(std::size_t channel, double gainDb)
    {
        m_outputGainParameters.setGain(channel, static_cast<T>(gainDb));
        pushOutputGainUpdate();
    }

    template<class T>
    void CudaSignalProcessor<T>::setOutputGains(const std::vector<double>& gainsDb)
    {
        m_inputGainParameters.setGains(std::vector<T>(gainsDb.begin(), gainsDb.end()));
        pushOutputGainUpdate();
    }

    template<class T>
    __global__ void processKernel(CudaSignalProcessorBuffers<T> buffers)
    {
        convertPcmToArray<T>(buffers.currentInputPcmFrame(),
            buffers.currentInputFrame(),
            buffers.frameSampleCount(),
            buffers.inputChannelCount(),
            buffers.inputFormat());
        __syncthreads();

        processEq(buffers.inputEqBuffers(),
            buffers.inputGainOutputFrames(),
            buffers.currentInputEqOutputFrame(),
            buffers.currentFrameIndex());
        __syncthreads();

        processEq(buffers.outputEqBuffers(),
            buffers.mixingOutputFrames(),
            buffers.currentOutputEqOutputFrame(),
            buffers.currentFrameIndex());
        __syncthreads();

        convertArrayToPcm<T>(buffers.currentInputFrame(),
            buffers.currentOutputPcmFrame(),
            buffers.frameSampleCount(),
            buffers.inputChannelCount(),
            buffers.outputFormat());
        __syncthreads();

        processSoundLevel(buffers.inputGainSoundLevelBuffers(), buffers.currentInputGainOutputFrame());
        processSoundLevel(buffers.inputEqSoundLevelBuffers(), buffers.currentInputEqOutputFrame());
        processSoundLevel(buffers.outputGainSoundLevelBuffers(), buffers.currentOutputFrame());
    }

    template<class T>
    const PcmAudioFrame& CudaSignalProcessor<T>::process(const PcmAudioFrame& inputFrame)
    {
        m_updateFunctionQueue.tryExecute();

        cudaMemcpy(m_buffers.currentInputPcmFrame(), inputFrame.data(), inputFrame.size(), cudaMemcpyHostToDevice);
        processKernel<<<1, 256>>>(m_buffers);
        cudaMemcpy(&m_outputFrame[0], m_buffers.currentOutputPcmFrame(), m_outputFrame.size(), cudaMemcpyDeviceToHost);

        notifySoundLevelUpdateIfNeeded();

        m_buffers.nextFrame();

        return m_outputFrame;
    }

    template<class T>
    void CudaSignalProcessor<T>::pushInputGainUpdate()
    {
        m_updateFunctionQueue.push([&]()
        {
            return m_inputGainParameters.tryApplyingUpdate([&]()
            {
                cudaMemcpy(m_buffers.inputGains(), m_inputGainParameters.gains().data(),
                    m_buffers.inputChannelCount() * sizeof(T), cudaMemcpyHostToDevice);
            });
        });
    }

    template<class T>
    void CudaSignalProcessor<T>::pushInputEqUpdate(std::size_t channel)
    {
        m_updateFunctionQueue.push([&, channel]()
        {
            return m_inputEqParameters.tryApplyingUpdate(channel, [&, channel]()
            {
                CudaEqBuffers<T>& eqBuffers = m_buffers.inputEqBuffers();

                cudaMemcpy(eqBuffers.biquadCoefficients(channel),
                    m_inputEqParameters.biquadCoefficients(channel).data(),
                    eqBuffers.filterCountPerChannel() * sizeof(BiquadCoefficients<T>),
                    cudaMemcpyHostToDevice);

                T d0 = m_inputEqParameters.d0(channel);
                cudaMemcpy(eqBuffers.d0() + channel, &d0, sizeof(T), cudaMemcpyHostToDevice);
            });
        });
    }

    template<class T>
    void CudaSignalProcessor<T>::pushMixingGainUpdate()
    {
        m_updateFunctionQueue.push([&]()
        {
            return m_mixingGainParameters.tryApplyingUpdate([&]()
            {
                cudaMemcpy(m_buffers.mixingGains(), m_mixingGainParameters.gains().data(),
                    m_buffers.mixingGainsSize() * sizeof(T), cudaMemcpyHostToDevice);
            });
        });
    }

    template<class T>
    void CudaSignalProcessor<T>::pushOutputEqUpdate(std::size_t channel)
    {
        m_updateFunctionQueue.push([&, channel]()
        {
            return m_outputEqParameters.tryApplyingUpdate(channel, [&, channel]()
            {
                CudaEqBuffers<T>& eqBuffers = m_buffers.outputEqBuffers();

                cudaMemcpy(eqBuffers.biquadCoefficients(channel),
                    m_outputEqParameters.biquadCoefficients(channel).data(),
                    eqBuffers.filterCountPerChannel() * sizeof(BiquadCoefficients<T>),
                    cudaMemcpyHostToDevice);

                T d0 = m_outputEqParameters.d0(channel);
                cudaMemcpy(eqBuffers.d0() + channel, &d0, sizeof(T), cudaMemcpyHostToDevice);
            });
        });
    }

    template<class T>
    void CudaSignalProcessor<T>::pushOutputGainUpdate()
    {
        m_updateFunctionQueue.push([&]()
        {
            return m_outputGainParameters.tryApplyingUpdate([&]()
            {
                cudaMemcpy(m_buffers.outputGains(), m_outputGainParameters.gains().data(),
                    m_buffers.outputChannelCount() * sizeof(T), cudaMemcpyHostToDevice);
            });
        });
    }

    template<class T>
    void CudaSignalProcessor<T>::notifySoundLevelUpdateIfNeeded()
    {
        m_frameSampleCounter += m_frameSampleCount;

        if (m_frameSampleCounter >= m_soundLevelLength)
        {
            m_buffers.inputGainSoundLevelBuffers().toVector(m_soundLevels[AnalysisDispatcher::SoundLevelType::InputGain]);
            m_buffers.inputEqSoundLevelBuffers().toVector(m_soundLevels[AnalysisDispatcher::SoundLevelType::InputEq]);
            m_buffers.outputGainSoundLevelBuffers().toVector(m_soundLevels[AnalysisDispatcher::SoundLevelType::OutputGain]);

            if (m_analysisDispatcher)
            {
                m_analysisDispatcher->notifySoundLevel(m_soundLevels);
            }
        }
    }
}

#endif
