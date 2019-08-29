#ifndef SIGNAL_PROCESSING_PARAMETERS_MIXING_PARAMETERS_H
#define SIGNAL_PROCESSING_PARAMETERS_MIXING_PARAMETERS_H

#include <SignalProcessing/Parameters/RealtimeParameters.h>

#include <Utils/ClassMacro.h>
#include <Utils/Exception/InvalidValueException.h>

#include <cstddef>
#include <vector>
#include <cmath>

namespace adaptone
{
    template<class T>
    class MixingParameters : public RealtimeParameters
    {
        std::size_t m_inputChannelCount;
        std::size_t m_outputChannelCount;
        std::vector<T> m_gains;

    public:
        MixingParameters(std::size_t inputChannelCount, std::size_t outputChannelCount);
        ~MixingParameters() override;

        DECLARE_NOT_COPYABLE(MixingParameters);
        DECLARE_NOT_MOVABLE(MixingParameters);

        void setGain(std::size_t inputChannel, std::size_t outputChannel, T gain);
        void setGains(std::size_t outputChannel, const std::vector<T>& gains);
        void setGains(const std::vector<T>& gains);
        const std::vector<T>& gains() const;
    };

    template<class T>
    MixingParameters<T>::MixingParameters(std::size_t inputChannelCount, std::size_t outputChannelCount) :
        m_inputChannelCount(inputChannelCount),
        m_outputChannelCount(outputChannelCount),
        m_gains(inputChannelCount * outputChannelCount, 0)
    {
    }

    template<class T>
    MixingParameters<T>::~MixingParameters()
    {
    }

    template<class T>
    void MixingParameters<T>::setGain(std::size_t inputChannel, std::size_t outputChannel, T gain)
    {
        if (inputChannel >= m_inputChannelCount || outputChannel >= m_outputChannelCount)
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid channel", "");
        }

        update([&]()
        {
            m_gains[outputChannel * m_inputChannelCount + inputChannel] = gain;
        });
    }

    template<class T>
    void MixingParameters<T>::setGains(std::size_t outputChannel, const std::vector<T>& gains)
    {
        if (outputChannel >= m_outputChannelCount)
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid output channel", "");
        }
        if (gains.size() != m_inputChannelCount)
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid channel count", "");
        }

        update([&]()
        {
            for (std::size_t i = 0; i < m_inputChannelCount; i++)
            {
                m_gains[outputChannel * m_inputChannelCount + i] = gains[i];
            }
        });
    }

    template<class T>
    void MixingParameters<T>::setGains(const std::vector<T>& gains)
    {
        if (gains.size() != m_gains.size())
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid channel count", "");
        }

        update([&]()
        {
            for (std::size_t i = 0; i < gains.size(); i++)
            {
                m_gains[i] = gains[i];
            }
        });
    }

    template<class T>
    const std::vector<T>& MixingParameters<T>::gains() const
    {
        return m_gains;
    }
}

#endif
