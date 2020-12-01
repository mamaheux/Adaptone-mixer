#ifndef SIGNAL_PROCESSING_CUDA_CUDA_SIGNAL_PROCESSOR_H
#define SIGNAL_PROCESSING_CUDA_CUDA_SIGNAL_PROCESSOR_H

#include <SignalProcessing/ProcessingDataType.h>
#include <SignalProcessing/SpecificSignalProcessor.h>
#include <SignalProcessing/Cuda/Buffers/CudaSignalProcessorBuffers.h>
#include <SignalProcessing/Cuda/Processing/EqProcessing.h>
#include <SignalProcessing/Cuda/Processing/GainProcessing.h>
#include <SignalProcessing/Cuda/Processing/MixProcessing.h>
#include <SignalProcessing/Cuda/Processing/DelayProcessing.h>
#include <SignalProcessing/Cuda/Processing/SoundLevelProcessing.h>
#include <SignalProcessing/Parameters/EqParameters.h>
#include <SignalProcessing/Parameters/GainParameters.h>
#include <SignalProcessing/Parameters/MixingParameters.h>
#include <SignalProcessing/Parameters/DelayParameters.h>
#include <SignalProcessing/AnalysisDispatcher.h>
#include <SignalProcessing/SignalProcessorParameters.h>

#include <Utils/ClassMacro.h>
#include <Utils/Data/PcmAudioFrame.h>
#include <Utils/Functional/FunctionQueue.h>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <memory>

namespace adaptone
{
    constexpr std::size_t CudaSignalProcessorFrameCount = 2;
    constexpr bool InitialParameterDirtyState = true;

    template<class T>
    class CudaSignalProcessor : public SpecificSignalProcessor
    {
        SignalProcessorParameters m_parameters;

        PcmAudioFrame m_outputFrame;
        CudaSignalProcessorBuffers<T> m_buffers;

        GainParameters<T> m_inputGainParameters;
        EqParameters<T> m_inputEqParameters;
        MixingParameters<T> m_mixingGainParameters;
        EqParameters<T> m_outputEqParameters;
        EqParameters<T> m_uniformizationEqParameters;
        GainParameters<T> m_outputGainParameters;
        DelayParameters m_outputDelayParameters;

        FunctionQueue<bool()> m_updateFunctionQueue;

        std::size_t m_frameSampleCounter;
        std::map<AnalysisDispatcher::SoundLevelType, std::vector<T>> m_soundLevels;
        std::shared_ptr<AnalysisDispatcher> m_analysisDispatcher;

    public:
        CudaSignalProcessor(std::shared_ptr<AnalysisDispatcher> analysisDispatcher,
            const SignalProcessorParameters& parameters);
        ~CudaSignalProcessor() override;

        DECLARE_NOT_COPYABLE(CudaSignalProcessor);
        DECLARE_NOT_MOVABLE(CudaSignalProcessor);

        void setInputGain(std::size_t channelIndex, double gain) override;
        void setInputGains(const std::vector<double>& gains) override;

        void setInputGraphicEqGains(std::size_t channelIndex, const std::vector<double>& gains) override;

        void setMixingGain(std::size_t inputChannelIndex, std::size_t outputChannelIndex, double gain) override;
        void setMixingGains(std::size_t outputChannelIndex, const std::vector<double>& gains) override;
        void setMixingGains(const std::vector<double>& gains) override;

        void setOutputGraphicEqGains(std::size_t channelIndex, const std::vector<double>& gains) override;
        void setOutputGraphicEqGains(std::size_t startChannelIndex, std::size_t n,
            const std::vector<double>& gains) override;

        void setOutputGain(std::size_t channelIndex, double gain) override;
        void setOutputGains(const std::vector<double>& gains) override;

        void setUniformizationGraphicEqGains(std::size_t channelIndex, const std::vector<double>& gains) override;
        void setUniformizationGraphicEqGains(std::size_t startChannelIndex, std::size_t n,
            const std::vector<double>& gains) override;

        void setOutputDelay(std::size_t channelIndex, std::size_t delay) override;
        void setOutputDelays(const std::vector<std::size_t>& delays) override;

        void forceRefreshParameters();

        const PcmAudioFrame& process(const PcmAudioFrame& inputFrame) override;

    private:
        void pushInputGainUpdate();
        void pushInputEqUpdate(std::size_t channelIndex);
        void pushMixingGainUpdate();
        void pushOutputEqUpdate(std::size_t channelIndex);
        void pushUniformizationEqUpdate(std::size_t channelIndex);
        void pushOutputGainUpdate();
        void pushOutputDelayUpdate();

        void notifySoundLevelIfNeeded();
        void notifyInputEqOutputFrame();
    };

    template<class T>
    CudaSignalProcessor<T>::CudaSignalProcessor(std::shared_ptr<AnalysisDispatcher> analysisDispatcher,
        const SignalProcessorParameters& parameters) :
        m_parameters(parameters),
        m_outputFrame(parameters.outputFormat(), parameters.outputChannelCount(), parameters.frameSampleCount()),
        m_buffers(parameters, CudaSignalProcessorFrameCount, 2 * parameters.eqCenterFrequencies().size()),
        m_inputGainParameters(parameters.inputChannelCount(), InitialParameterDirtyState),
        m_inputEqParameters(parameters.sampleFrequency(),
            parameters.eqCenterFrequencies(),
            parameters.inputChannelCount(),
            InitialParameterDirtyState),
        m_mixingGainParameters(parameters.inputChannelCount(),
            parameters.outputChannelCount(),
            InitialParameterDirtyState),
        m_outputEqParameters(parameters.sampleFrequency(),
            parameters.eqCenterFrequencies(),
            parameters.outputChannelCount(),
            InitialParameterDirtyState),
        m_uniformizationEqParameters(parameters.sampleFrequency(),
            parameters.eqCenterFrequencies(),
            parameters.outputChannelCount(),
            InitialParameterDirtyState),
        m_outputGainParameters(parameters.outputChannelCount(), InitialParameterDirtyState),
        m_outputDelayParameters(parameters.outputChannelCount(),
            parameters.maxOutputDelay(),
            InitialParameterDirtyState),

        m_frameSampleCounter(0),
        m_analysisDispatcher(move(analysisDispatcher))
    {
        m_soundLevels[AnalysisDispatcher::SoundLevelType::InputGain] = std::vector<T>(parameters.inputChannelCount());
        m_soundLevels[AnalysisDispatcher::SoundLevelType::InputEq] = std::vector<T>(parameters.inputChannelCount());
        m_soundLevels[AnalysisDispatcher::SoundLevelType::OutputGain] = std::vector<T>(parameters.outputChannelCount());

        pushInputGainUpdate();
        pushMixingGainUpdate();
        pushOutputGainUpdate();
        pushOutputDelayUpdate();

        for (std::size_t i = 0; i < parameters.inputChannelCount(); i++)
        {
            pushInputEqUpdate(i);
        }

        for (std::size_t i = 0; i < parameters.outputChannelCount(); i++)
        {
            pushOutputEqUpdate(i);
            pushUniformizationEqUpdate(i);
        }

        forceRefreshParameters();
    }

    template<class T>
    CudaSignalProcessor<T>::~CudaSignalProcessor()
    {
    }

    template<class T>
    void CudaSignalProcessor<T>::setInputGain(std::size_t channelIndex, double gain)
    {
        m_inputGainParameters.setGain(channelIndex, static_cast<T>(gain));
        pushInputGainUpdate();
    }

    template<class T>
    void CudaSignalProcessor<T>::setInputGains(const std::vector<double>& gains)
    {
        m_inputGainParameters.setGains(std::vector<T>(gains.begin(), gains.end()));
        pushInputGainUpdate();
    }

    template<class T>
    void CudaSignalProcessor<T>::setInputGraphicEqGains(std::size_t channelIndex, const std::vector<double>& gains)
    {
        m_inputEqParameters.setGraphicEqGains(channelIndex, gains);
        pushInputEqUpdate(channelIndex);
    }

    template<class T>
    void CudaSignalProcessor<T>::setMixingGain(std::size_t inputChannelIndex, std::size_t outputChannelIndex,
        double gain)
    {
        m_mixingGainParameters.setGain(inputChannelIndex, outputChannelIndex, static_cast<T>(gain));
        pushMixingGainUpdate();
    }

    template<class T>
    void CudaSignalProcessor<T>::setMixingGains(std::size_t outputChannelIndex, const std::vector<double>& gains)
    {
        m_mixingGainParameters.setGains(outputChannelIndex, std::vector<T>(gains.begin(), gains.end()));
        pushMixingGainUpdate();
    }

    template<class T>
    void CudaSignalProcessor<T>::setMixingGains(const std::vector<double>& gains)
    {
        m_mixingGainParameters.setGains(std::vector<T>(gains.begin(), gains.end()));
        pushMixingGainUpdate();
    }

    template<class T>
    void CudaSignalProcessor<T>::setOutputGraphicEqGains(std::size_t channelIndex, const std::vector<double>& gains)
    {
        m_outputEqParameters.setGraphicEqGains(channelIndex, gains);
        pushOutputEqUpdate(channelIndex);
    }

    template<class T>
    void CudaSignalProcessor<T>::setOutputGraphicEqGains(std::size_t startChannelIndex, std::size_t n,
        const std::vector<double>& gains)
    {
        m_outputEqParameters.setGraphicEqGains(startChannelIndex, n, gains);

        for (std::size_t i = 0; i < n; i++)
        {
            pushOutputEqUpdate(startChannelIndex + i);
        }
    }

    template<class T>
    void CudaSignalProcessor<T>::setOutputGain(std::size_t channelIndex, double gain)
    {
        m_outputGainParameters.setGain(channelIndex, static_cast<T>(gain));
        pushOutputGainUpdate();
    }

    template<class T>
    void CudaSignalProcessor<T>::setOutputGains(const std::vector<double>& gains)
    {
        m_outputGainParameters.setGains(std::vector<T>(gains.begin(), gains.end()));
        pushOutputGainUpdate();
    }

    template<class T>
    void CudaSignalProcessor<T>::setUniformizationGraphicEqGains(std::size_t channelIndex,
        const std::vector<double>& gains)
    {
        m_uniformizationEqParameters.setGraphicEqGains(channelIndex, gains);
        pushUniformizationEqUpdate(channelIndex);
    }

    template<class T>
    void CudaSignalProcessor<T>::setUniformizationGraphicEqGains(std::size_t startChannelIndex, std::size_t n,
            const std::vector<double>& gains)
    {
        m_uniformizationEqParameters.setGraphicEqGains(startChannelIndex, n, gains);

        for (std::size_t i = 0; i < n; i++)
        {
            pushUniformizationEqUpdate(startChannelIndex + i);
        }
    }

    template<class T>
    void CudaSignalProcessor<T>::setOutputDelay(std::size_t channelIndex, std::size_t delay)
    {
        m_outputDelayParameters.setDelay(channelIndex, delay);
        pushOutputDelayUpdate();
    }

    template<class T>
    void CudaSignalProcessor<T>::setOutputDelays(const std::vector<std::size_t>& delays)
    {
        m_outputDelayParameters.setDelays(delays);
        pushOutputDelayUpdate();
    }

    template<class T>
    void CudaSignalProcessor<T>::forceRefreshParameters()
    {
        while (m_updateFunctionQueue.size() > 0)
        {
            m_updateFunctionQueue.execute();
        }
    }

    template<class T>
    __global__ void processKernel(CudaSignalProcessorBuffers<T> buffers)
    {
        cooperative_groups::grid_group grid = cooperative_groups::this_grid();
        convertPcmToArray<T>(buffers.currentInputPcmFrame(),
            buffers.currentInputFrame(),
            buffers.frameSampleCount(),
            buffers.inputChannelCount(),
            buffers.inputFormat());
        grid.sync();

        processGain(buffers.currentInputFrame(),
            buffers.currentInputGainOutputFrame(),
            buffers.inputGains(),
            buffers.frameSampleCount(),
            buffers.inputChannelCount());
        grid.sync();

        processEq(buffers.inputEqBuffers(),
            buffers.inputGainOutputFrames(),
            buffers.currentInputEqOutputFrame(),
            buffers.currentFrameIndex());
        grid.sync();

        processMix(buffers.currentInputEqOutputFrame(),
            buffers.currentMixingOutputFrame(),
            buffers.mixingGains(),
            buffers.frameSampleCount(),
            buffers.inputChannelCount(),
            buffers.outputChannelCount());
        grid.sync();

        processEq(buffers.outputEqBuffers(),
            buffers.mixingOutputFrames(),
            buffers.currentOutputEqOutputFrame(),
            buffers.currentFrameIndex());
        grid.sync();

        processEq(buffers.uniformizationEqBuffers(),
            buffers.outputEqOutputFrames(),
            buffers.currentUniformizationEqOutputFrame(),
            buffers.currentFrameIndex());
        grid.sync();

        processGain(buffers.currentUniformizationEqOutputFrame(),
            buffers.currentOutputFrame(),
            buffers.outputGains(),
            buffers.frameSampleCount(),
            buffers.outputChannelCount());
        grid.sync();

        processDelay(buffers.currentOutputFrame(),
            buffers.delayedOutputFrames(),
            buffers.outputDelays(),
            buffers.frameSampleCount(),
            buffers.outputChannelCount(),
            buffers.currentDelayedOutputFrameIndex(),
            buffers.delayedOutputFrameCount());
        grid.sync();

        convertArrayToPcm<T>(buffers.currentDelayedOutputFrame(),
            buffers.currentOutputPcmFrame(),
            buffers.frameSampleCount(),
            buffers.outputChannelCount(),
            buffers.outputFormat());
        grid.sync();

        processSoundLevel(buffers.inputGainSoundLevelBuffers(), buffers.currentInputGainOutputFrame());
        processSoundLevel(buffers.inputEqSoundLevelBuffers(), buffers.currentInputEqOutputFrame());
        processSoundLevel(buffers.outputGainSoundLevelBuffers(), buffers.currentOutputFrame());
    }

    template<class T>
    void launchProcessKernel(CudaSignalProcessorBuffers<T>& buffers)
    {
        void* args[] = { &buffers };
        dim3 gridDim(2);
        dim3 blockDim(256);
        cudaLaunchCooperativeKernel(reinterpret_cast<void*>(processKernel<T>), gridDim, blockDim, args);
    }

    template<class T>
    const PcmAudioFrame& CudaSignalProcessor<T>::process(const PcmAudioFrame& inputFrame)
    {
        m_updateFunctionQueue.tryExecute();

        m_buffers.copyInputFrame(inputFrame);
        launchProcessKernel(m_buffers);
        m_buffers.copyOutputFrame(m_outputFrame);

        notifySoundLevelIfNeeded();
        notifyInputEqOutputFrame();

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
                m_buffers.updateInputGains(m_inputGainParameters.gains().data());
            });
        });
    }

    template<class T>
    void CudaSignalProcessor<T>::pushInputEqUpdate(std::size_t channelIndex)
    {
        m_updateFunctionQueue.push([&, channelIndex]()
        {
            return m_inputEqParameters.tryApplyingUpdate(channelIndex, [&, channelIndex]()
            {
                m_buffers.inputEqBuffers().update(channelIndex,
                    m_inputEqParameters.biquadCoefficients(channelIndex).data(),
                    m_inputEqParameters.d0(channelIndex));
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
                m_buffers.updateMixingGains(m_mixingGainParameters.gains().data());
            });
        });
    }

    template<class T>
    void CudaSignalProcessor<T>::pushOutputEqUpdate(std::size_t channelIndex)
    {
        m_updateFunctionQueue.push([&, channelIndex]()
        {
            return m_outputEqParameters.tryApplyingUpdate(channelIndex, [&, channelIndex]()
            {
                m_buffers.outputEqBuffers().update(channelIndex,
                    m_outputEqParameters.biquadCoefficients(channelIndex).data(),
                    m_outputEqParameters.d0(channelIndex));
            });
        });
    }

    template<class T>
    void CudaSignalProcessor<T>::pushUniformizationEqUpdate(std::size_t channelIndex)
    {
        m_updateFunctionQueue.push([&, channelIndex]()
        {
            return m_uniformizationEqParameters.tryApplyingUpdate(channelIndex, [&, channelIndex]()
            {
                m_buffers.uniformizationEqBuffers().update(channelIndex,
                    m_uniformizationEqParameters.biquadCoefficients(channelIndex).data(),
                    m_uniformizationEqParameters.d0(channelIndex));
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
                m_buffers.updateOutputGains(m_outputGainParameters.gains().data());
            });
        });
    }

    template<class T>
    void CudaSignalProcessor<T>::pushOutputDelayUpdate()
    {
        m_updateFunctionQueue.push([&]()
        {
            return m_outputDelayParameters.tryApplyingUpdate([&]()
            {
               m_buffers.updateOutputDelays(m_outputDelayParameters.delays().data());
            });
        });
    }

    template<class T>
    void CudaSignalProcessor<T>::notifySoundLevelIfNeeded()
    {
        m_frameSampleCounter += m_parameters.frameSampleCount();

        if (m_frameSampleCounter >= m_parameters.soundLevelLength())
        {
            m_frameSampleCounter = 0;

            m_buffers.inputGainSoundLevelBuffers().toVector(m_soundLevels[AnalysisDispatcher::SoundLevelType::InputGain]);
            m_buffers.inputEqSoundLevelBuffers().toVector(m_soundLevels[AnalysisDispatcher::SoundLevelType::InputEq]);
            m_buffers.outputGainSoundLevelBuffers().toVector(m_soundLevels[AnalysisDispatcher::SoundLevelType::OutputGain]);

            m_buffers.inputGainSoundLevelBuffers().resetBuffer();
            m_buffers.inputEqSoundLevelBuffers().resetBuffer();
            m_buffers.outputGainSoundLevelBuffers().resetBuffer();

            if (m_analysisDispatcher)
            {
                m_analysisDispatcher->notifySoundLevel(m_soundLevels);
            }
        }
    }

    template<class T>
    void CudaSignalProcessor<T>::notifyInputEqOutputFrame()
    {
        if (m_analysisDispatcher)
        {
            m_analysisDispatcher->notifyInputEqOutputFrame([&](T* buffer)
            {
                m_buffers.copyCurrentInputEqOutputFrame(buffer);
            });
        }
    }
}

#endif
