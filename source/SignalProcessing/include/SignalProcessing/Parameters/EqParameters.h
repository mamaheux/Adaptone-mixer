#ifndef SIGNAL_PROCESSING_PARAMETERS_EQ_PARAMETERS_H
#define SIGNAL_PROCESSING_PARAMETERS_EQ_PARAMETERS_H

#include <SignalProcessing/Filters/Design/GraphicEqDesigner.h>
#include <SignalProcessing/Parameters/RealtimeParameters.h>

#include <Utils/ClassMacro.h>
#include <Utils/Exception/InvalidValueException.h>

#include <deque>
#include <vector>

namespace adaptone
{
    template<class T>
    class EqParameters
    {
        std::deque<RealtimeParameters> m_realtimeParameters;

        std::vector<double> m_eqCenterFrequencies;

        std::deque<GraphicEqDesigner<T>> m_graphicEqDesigners;

    public:
        EqParameters(std::size_t sampleFrequency, const std::vector<double>& eqCenterFrequencies,
            std::size_t channelCount);
        virtual ~EqParameters();

        DECLARE_NOT_COPYABLE(EqParameters);
        DECLARE_NOT_MOVABLE(EqParameters);

        void setGraphicEqGains(std::size_t channel, const std::vector<double>& gainsDb);
        void setGraphicEqGains(std::size_t startChannelIndex, std::size_t n, const std::vector<double>& gainsDb);

        const std::vector<BiquadCoefficients<T>>& biquadCoefficients(std::size_t channel) const;
        const T d0(std::size_t channel) const;

        bool isDirty(std::size_t channel);

        void applyUpdate(std::size_t channel, const std::function<void()>& function);
        bool tryApplyingUpdate(std::size_t channel, const std::function<void()>& function);
    };

    template<class T>
    EqParameters<T>::EqParameters(std::size_t sampleFrequency, const std::vector<double>& eqCenterFrequencies,
        std::size_t channelCount) :
        m_eqCenterFrequencies(eqCenterFrequencies)
    {
        for (std::size_t i = 0; i < channelCount; i++)
        {
            m_graphicEqDesigners.emplace_back(sampleFrequency, eqCenterFrequencies);
            m_realtimeParameters.emplace_back();
        }
    }

    template<class T>
    EqParameters<T>::~EqParameters()
    {
    }

    template<class T>
    void EqParameters<T>::setGraphicEqGains(std::size_t channel, const std::vector<double>& gainsDb)
    {
        if (channel >= m_realtimeParameters.size())
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid channel", "");
        }

        m_realtimeParameters[channel].update([&]()
        {
            m_graphicEqDesigners[channel].update(gainsDb);
        });
    }

    template<class T>
    void EqParameters<T>::setGraphicEqGains(std::size_t startChannelIndex, std::size_t n,
        const std::vector<double>& gainsDb)
    {
        if (startChannelIndex + n > m_realtimeParameters.size())
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid channels", "");
        }

        if (n > 0)
        {
            m_realtimeParameters[startChannelIndex].update([&]()
            {
                m_graphicEqDesigners[startChannelIndex].update(gainsDb);
            });

            for (std::size_t i = 1; i < n; i++)
            {
                m_realtimeParameters[startChannelIndex + i].update([&]()
                {
                    m_graphicEqDesigners[startChannelIndex + i]
                        .update(m_graphicEqDesigners[startChannelIndex].biquadCoefficients(),
                            m_graphicEqDesigners[startChannelIndex].d0());
                });
            }
        }
    }

    template<class T>
    const std::vector<BiquadCoefficients<T>>& EqParameters<T>::biquadCoefficients(std::size_t channel) const
    {
        if (channel >= m_realtimeParameters.size())
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid channel", "");
        }

        return m_graphicEqDesigners[channel].biquadCoefficients();
    }

    template<class T>
    const T EqParameters<T>::d0(std::size_t channel) const
    {
        if (channel >= m_graphicEqDesigners.size())
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid channel", "");
        }

        return m_graphicEqDesigners[channel].d0();
    }

    template<class T>
    bool EqParameters<T>::isDirty(std::size_t channel)
    {
        if (channel >= m_realtimeParameters.size())
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid channel", "");
        }

        return m_realtimeParameters[channel].isDirty();
    }

    template<class T>
    void EqParameters<T>::applyUpdate(std::size_t channel, const std::function<void()>& function)
    {
        if (channel >= m_realtimeParameters.size())
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid channel", "");
        }

        m_realtimeParameters[channel].applyUpdate(function);
    }

    template<class T>
    bool EqParameters<T>::tryApplyingUpdate(std::size_t channel, const std::function<void()>& function)
    {
        if (channel >= m_realtimeParameters.size())
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid channel", "");
        }

        return m_realtimeParameters[channel].tryApplyingUpdate(function);
    }
}

#endif
