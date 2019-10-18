#ifndef SIGNAL_PROCESSING_PARAMETERS_GAIN_PARAMETERS_H
#define SIGNAL_PROCESSING_PARAMETERS_GAIN_PARAMETERS_H

#include <SignalProcessing/Parameters/RealtimeParameters.h>

#include <Utils/ClassMacro.h>
#include <Utils/Exception/InvalidValueException.h>

#include <cstddef>
#include <vector>

namespace adaptone
{
    template<class T>
    class GainParameters : public RealtimeParameters
    {
        std::vector<T> m_gains;

    public:
        GainParameters(std::size_t channelCount, bool isDirty = false);
        ~GainParameters() override;

        DECLARE_NOT_COPYABLE(GainParameters);
        DECLARE_NOT_MOVABLE(GainParameters);

        void setGain(std::size_t channel, T gain);
        void setGains(const std::vector<T>& gains);
        const std::vector<T>& gains() const;
    };

    template<class T>
    GainParameters<T>::GainParameters(std::size_t channelCount, bool isDirty) :
        RealtimeParameters(isDirty), m_gains(channelCount, 1)
    {
    }

    template<class T>
    GainParameters<T>::~GainParameters()
    {
    }

    template<class T>
    void GainParameters<T>::setGain(std::size_t channel, T gain)
    {
        if (channel >= m_gains.size())
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid channel", "");
        }

        update([&]()
        {
            m_gains[channel] = gain;
        });
    }

    template<class T>
    void GainParameters<T>::setGains(const std::vector<T>& gains)
    {
        if (gains.size() != m_gains.size())
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid channel count", "");
        }

        update([&]()
        {
            m_gains = gains;
        });
    }

    template<class T>
    const std::vector<T>& GainParameters<T>::gains() const
    {
        return m_gains;
    }
}

#endif
