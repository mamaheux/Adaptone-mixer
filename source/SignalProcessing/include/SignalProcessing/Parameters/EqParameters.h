#ifndef SIGNAL_PROCESSING_PARAMETERS_EQ_PARAMETERS_H
#define SIGNAL_PROCESSING_PARAMETERS_EQ_PARAMETERS_H

#include <SignalProcessing/Filters/Design/GraphicEqDesigner.h>
#include <SignalProcessing/Filters/Design/ParametricEqDesigner.h>
#include <SignalProcessing/Parameters/RealtimeParameters.h>

#include <Utils/ClassMacro.h>
#include <Utils/Exception/InvalidValueException.h>

#include <deque>

namespace adaptone
{
    template<class T>
    class EqParameters : public RealtimeParameters
    {
        std::vector<double> m_eqCenterFrequencies;

        std::deque<GraphicEqDesigner<T>> m_graphicEqDesigners;
        ParametricEqDesigner<T> m_parametricEqDesigner;

    public:
        EqParameters(std::size_t sampleFrequency, std::size_t parametricFilterCount,
            const std::vector<double>& eqCenterFrequencies, std::size_t channelCount);
        ~EqParameters() override;

        DECLARE_NOT_COPYABLE(EqParameters);
        DECLARE_NOT_MOVABLE(EqParameters);

        void setParametricEqParameters(std::size_t channel, const std::vector<ParametricEqParameters>& parameters);
        void setGraphicEqGains(std::size_t channel, const std::vector<double>& gainsDb);

        const std::vector<BiquadCoefficients<T>>& biquadCoefficients(std::size_t channel) const;
        const T d0(std::size_t channel) const;
    };

    template<class T>
    EqParameters<T>::EqParameters(std::size_t sampleFrequency, std::size_t parametricFilterCount,
        const std::vector<double>& eqCenterFrequencies, std::size_t channelCount) :
        m_eqCenterFrequencies(eqCenterFrequencies),
        m_parametricEqDesigner(parametricFilterCount, sampleFrequency)
    {
        for (std::size_t i = 0; i < channelCount; i++)
        {
            m_graphicEqDesigners.emplace_back(sampleFrequency, eqCenterFrequencies);
        }
    }

    template<class T>
    EqParameters<T>::~EqParameters()
    {
    }

    template<class T>
    void EqParameters<T>::setParametricEqParameters(std::size_t channel,
        const std::vector<ParametricEqParameters>& parameters)
    {
        if (channel >= m_graphicEqDesigners.size())
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid channel", "");
        }

        update([&]()
        {
            m_parametricEqDesigner.update(parameters);
            m_graphicEqDesigners[channel].update(m_parametricEqDesigner.gainsDb(m_eqCenterFrequencies));
        });
    }

    template<class T>
    void EqParameters<T>::setGraphicEqGains(std::size_t channel, const std::vector<double>& gainsDb)
    {
        if (channel >= m_graphicEqDesigners.size())
        {
            THROW_INVALID_VALUE_EXCEPTION("Invalid channel", "");
        }

        update([&]()
        {
            m_graphicEqDesigners[channel].update(gainsDb);
        });
    }

    template<class T>
    const std::vector<BiquadCoefficients<T>>& EqParameters<T>::biquadCoefficients(std::size_t channel) const
    {
        if (channel >= m_graphicEqDesigners.size())
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
}

#endif
