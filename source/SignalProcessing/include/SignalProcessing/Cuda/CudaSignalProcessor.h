#ifndef SIGNAL_PROCESSING_CUDA_CUDA_SIGNAL_PROCESSOR_H
#define SIGNAL_PROCESSING_CUDA_CUDA_SIGNAL_PROCESSOR_H

#include <SignalProcessing/ProcessingDataType.h>
#include <SignalProcessing/SpecificSignalProcessor.h>
#include <SignalProcessing/Cuda/CudaSignalProcessorBuffers.h>
#include <SignalProcessing/Parameters/EqParameters.h>
#include <SignalProcessing/Parameters/GainParameters.h>
#include <SignalProcessing/Parameters/MixingParameters.h>

#include <Utils/ClassMacro.h>
#include <Utils/Data/PcmAudioFrame.h>
#include <Utils/Functional/FunctionQueue.h>

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

        GainParameters<T> m_inputGainParameters;
        EqParameters<T> m_inputEqParameters;
        MixingParameters<T> m_mixingGainParameters;
        EqParameters<T> m_outputEqParameters;
        GainParameters<T> m_outputGainParameters;

        FunctionQueue<bool()> m_updateFunctionQueue;

    public:
        CudaSignalProcessor(std::size_t frameSampleCount,
            std::size_t sampleFrequency,
            std::size_t inputChannelCount,
            std::size_t outputChannelCount,
            PcmAudioFrame::Format inputFormat,
            PcmAudioFrame::Format outputFormat,
            std::size_t parametricEqFilterCount,
            const std::vector<double>& eqCenterFrequencies);
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
    };

    template<class T>
    CudaSignalProcessor<T>::CudaSignalProcessor(size_t frameSampleCount,
        size_t sampleFrequency,
        size_t inputChannelCount,
        size_t outputChannelCount,
        PcmAudioFrame::Format inputFormat,
        PcmAudioFrame::Format outputFormat,
        std::size_t parametricEqFilterCount,
        const std::vector<double>& eqCenterFrequencies) :
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
        m_outputGainParameters(outputChannelCount)
    {
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
        m_updateFunctionQueue.tryExecute();

        cudaMemcpy(m_buffers.currentInputPcmFrame(), inputFrame.data(), inputFrame.size(), cudaMemcpyHostToDevice);
        processKernel<<<1, 256>>>(m_buffers);
        cudaMemcpy(&m_outputFrame[0], m_buffers.currentOutputPcmFrame(), m_outputFrame.size(), cudaMemcpyDeviceToHost);

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

            return m_inputEqParameters.tryApplyingUpdate([&, channel]()
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
            return m_outputEqParameters.tryApplyingUpdate([&, channel]()
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
}

#endif
