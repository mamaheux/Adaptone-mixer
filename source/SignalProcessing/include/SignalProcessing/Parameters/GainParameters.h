#ifndef SIGNAL_PROCESSING_PARAMETERS_GAIN_PARAMETERS_H
#define SIGNAL_PROCESSING_PARAMETERS_GAIN_PARAMETERS_H

#include <SignalProcessing/Parameters/RealtimeParameters.h>

#include <Utils/ClassMacro.h>
#include <Utils/Exception/InvalidValueException.h>

#include <cstddef>
#include <vector>
#include <cmath>

namespace adaptone
{
    template<class T>
    class GainParameters : public RealtimeParameters
    {
        std::vector<T> m_gains;

    public:
        GainParameters(std::size_t channelCount);
        ~GainParameters() override;

        DECLARE_NOT_COPYABLE(GainParameters);
        DECLARE_NOT_MOVABLE(GainParameters);

        void setGain(std::size_t channel, T gainDb);
        void setGains(const std::vector<T>& gainsDb);
        const std::vector<T>& gains() const;
    };

    template<class T>
    GainParameters<T>::GainParameters(std::size_t channelCount) : m_gains(channelCount, 1)
    {
    }

    template<class T>
    GainParameters<T>::~GainParameters()
    {
    }

    template<class T>
    void GainParameters<T>::setGain(std::size_t channel, T gainDb)
    {
        if (channel >= m_gains.size())
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid channel", "");
        }

        uptate([&]()
        {
            m_gains[channel] = std::pow(10, gainDb / 20);
        });
    }

    template<class T>
    void GainParameters<T>::setGains(const std::vector<T>& gainsDb)
    {
        if (gainsDb.size() != m_gains.size())
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid channel count", "");
        }

        uptate([&]()
        {
            for (std::size_t i = 0; i < gainsDb.size(); i++)
            {
                m_gains[i] = std::pow(10, gainsDb[i] / 20);
            }
        });
    }

    template<class T>
    const std::vector<T>& GainParameters<T>::gains() const
    {
        return m_gains;
    }
}

#endif
