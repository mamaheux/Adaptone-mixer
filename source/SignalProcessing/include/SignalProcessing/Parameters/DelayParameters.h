#ifndef SIGNAL_PROCESSING_PARAMETERS_DELAY_PARAMETERS_H
#define SIGNAL_PROCESSING_PARAMETERS_DELAY_PARAMETERS_H

#include <SignalProcessing/Parameters/RealtimeParameters.h>

#include <Utils/ClassMacro.h>
#include <Utils/Exception/InvalidValueException.h>

#include <cstddef>
#include <vector>

namespace adaptone
{
    class DelayParameters : public RealtimeParameters
    {
        std::size_t m_maxDelay;
        std::vector<std::size_t> m_delays;

    public:
        DelayParameters(std::size_t channelCount, std::size_t maxDelay);
        ~DelayParameters() override;

        DECLARE_NOT_COPYABLE(DelayParameters);
        DECLARE_NOT_MOVABLE(DelayParameters);

        void setDelay(std::size_t channel, std::size_t delay);
        void setDelays(const std::vector<std::size_t>& delays);
        const std::vector<std::size_t>& delays() const;
    };

    inline void DelayParameters::setDelay(std::size_t channel, std::size_t delay)
    {
        if (channel >= m_delays.size())
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid channel", "");
        }
        if (delay > m_maxDelay)
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid delay", "");
        }

        update([&]()
        {
            m_delays[channel] = delay;
        });
    }

    inline void DelayParameters::setDelays(const std::vector<std::size_t>& delays)
    {
        if (delays.size() != m_delays.size())
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid channel count", "");
        }
        for (std::size_t delay : delays)
        {
            if (delay > m_maxDelay)
            {
                THROW_INVALID_VALUE_EXCEPTION("Invalid delay", "");
            }
        }

        update([&]()
        {
            m_delays = delays;
        });
    }

    inline const std::vector<std::size_t>& DelayParameters::delays() const
    {
        return m_delays;
    }
}

#endif
