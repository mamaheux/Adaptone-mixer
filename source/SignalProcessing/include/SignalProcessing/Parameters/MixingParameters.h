#ifndef SIGNAL_PROCESSING_PARAMETERS_MIXING_PARAMETERS_H
#define SIGNAL_PROCESSING_PARAMETERS_MIXING_PARAMETERS_H

#include <Utils/ClassMacro.h>
#include <Utils/Exception/InvalidValueException.h>

#include <cstddef>
#include <vector>
#include <cmath>

namespace adaptone
{
    template<class T>
    class MixingParameters
    {
        std::size_t m_inputChannelCount;
        std::size_t m_outputChannelCount;
        std::vector<T> m_gains;

    public:
        MixingParameters(std::size_t inputChannelCount, std::size_t outputChannelCount);
        virtual ~MixingParameters();

        DECLARE_NOT_COPYABLE(MixingParameters);
        DECLARE_NOT_MOVABLE(MixingParameters);

        void setGain(std::size_t inputChannel, std::size_t outputChannel, T gainDb);
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
    void MixingParameters<T>::setGain(std::size_t inputChannel, std::size_t outputChannel, T gainDb)
    {
        if (inputChannel >= m_inputChannelCount || outputChannel >= m_outputChannelCount)
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid channel", "");
        }

        m_gains[outputChannel * m_inputChannelCount + inputChannel] = std::pow(10, gainDb / 20);
    }

    template<class T>
    const std::vector<T>& MixingParameters<T>::gains() const
    {
        return m_gains;
    }
}

#endif
