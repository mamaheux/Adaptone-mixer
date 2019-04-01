#ifndef SIGNAL_PROCESSING_FILTERS_DESIGN_PARAMETRIC_EQ_DESIGNER_H
#define SIGNAL_PROCESSING_FILTERS_DESIGN_PARAMETRIC_EQ_DESIGNER_H

#include <SignalProcessing/Filters/BiquadCoefficients.h>

#include <Utils/ClassMacro.h>
#include <Utils/Exception/InvalidValueException.h>

#include <armadillo>

#include <vector>
#include <cmath>

namespace adaptone
{
    struct ParametricEqParameters
    {
        double cutoffFrequency;
        double Q;
        double gainDb;

        ParametricEqParameters(double cutoffFrequency, double Q, double gainDb);
        virtual ~ParametricEqParameters();
    };

    template<class T>
    class ParametricEqDesigner
    {
        std::size_t m_filterCount;
        double m_sampleFrequency;

        std::vector<BiquadCoefficients<T>> m_biquadCoefficients;

    public:
        ParametricEqDesigner(std::size_t filterCount, std::size_t sampleFrequency);
        virtual ~ParametricEqDesigner();

        DECLARE_NOT_COPYABLE(ParametricEqDesigner);
        DECLARE_NOT_MOVABLE(ParametricEqDesigner);

        void update(const std::vector<ParametricEqParameters>& parameters);
        const std::vector<BiquadCoefficients<T>>& biquadCoefficients() const;
        std::vector<double> gainsDb(const std::vector<double>& frequencies) const;

    private:
        void designLowShelvingFilter(BiquadCoefficients<T>& coefficients, const ParametricEqParameters& parameter);
        void designHighShelvingFilter(BiquadCoefficients<T>& coefficients, const ParametricEqParameters& parameter);
        void designPeakFilter(BiquadCoefficients<T>& coefficients, const ParametricEqParameters& parameter);
    };

    inline ParametricEqParameters::ParametricEqParameters(double cutoffFrequency, double Q, double gainDb) :
        cutoffFrequency(cutoffFrequency), Q(Q), gainDb(gainDb)
    {
    }

    inline ParametricEqParameters::~ParametricEqParameters()
    {
    }

    template<class T>
    inline ParametricEqDesigner<T>::ParametricEqDesigner(std::size_t filterCount, std::size_t sampleFrequency) :
        m_filterCount(filterCount), m_sampleFrequency(sampleFrequency), m_biquadCoefficients(filterCount)
    {
        if (m_filterCount < 2)
        {
            THROW_INVALID_VALUE_EXCEPTION("filterCount", "");
        }
    }

    template<class T>
    inline ParametricEqDesigner<T>::~ParametricEqDesigner()
    {
    }

    template<class T>
    inline void ParametricEqDesigner<T>::update(const std::vector<ParametricEqParameters>& parameters)
    {
        if (m_filterCount != parameters.size())
        {
            THROW_INVALID_VALUE_EXCEPTION("parameters.size()", "");
        }

        designLowShelvingFilter(m_biquadCoefficients[0], parameters[0]);
        designHighShelvingFilter(m_biquadCoefficients[m_filterCount - 1], parameters[m_filterCount - 1]);

        for (std::size_t i = 1; i < m_filterCount - 1; i++)
        {
            designPeakFilter(m_biquadCoefficients[i], parameters[i]);
        }
    }

    template<class T>
    inline const std::vector<BiquadCoefficients<T>>& ParametricEqDesigner<T>::biquadCoefficients() const
    {
        return m_biquadCoefficients;
    }

    template<class T>
    inline std::vector<double> ParametricEqDesigner<T>::gainsDb(const std::vector<double>& frequencies) const
    {
        arma::vec w(frequencies);
        w = 2 * M_PI * w / m_sampleFrequency;

        std::complex<double> j(0, 1);
        arma::cx_vec jw = -j * w;
        arma::cx_vec jw2 = 2 * jw;

        arma::cx_vec h(frequencies.size());
        h.ones();

        for (const BiquadCoefficients<T>& c : m_biquadCoefficients)
        {
            h %= (c.b0 + c.b1 * arma::exp(jw) + c.b2 * arma::exp(jw2)) /
                (1 + c.a1 * arma::exp(jw) + c.a2 * arma::exp(jw2));
        }

        return arma::conv_to<std::vector<double>>::from(20 * arma::log10(arma::abs(h)));
    }

    template<class T>
    inline void ParametricEqDesigner<T>::designLowShelvingFilter(BiquadCoefficients<T>& coefficients,
        const ParametricEqParameters& parameter)
    {
        double k = std::tan((M_PI * parameter.cutoffFrequency) / m_sampleFrequency);
        double v0 = std::pow(10.0, parameter.gainDb / 20.0);
        double root2 = 1.0 / parameter.Q;

        if (v0 < 1)
        {
            v0 = 1.0 / v0;
        }

        if (parameter.gainDb > 0)
        {
            coefficients.b0 = static_cast<T>((1 + std::sqrt(v0) * root2 * k + v0 * k * k) / (1 + root2 * k + k * k));
            coefficients.b1 = static_cast<T>((2 * (v0 * k * k - 1)) / (1 + root2 * k + k * k));
            coefficients.b2 = static_cast<T>((1 - std::sqrt(v0) * root2 * k + v0 * k * k) / (1 + root2 * k + k * k));
            coefficients.a1 = static_cast<T>((2 * (k * k - 1)) / (1 + root2 * k + k * k));
            coefficients.a2 = static_cast<T>((1 - root2 * k + k * k) / (1 + root2 * k + k * k));
        }
        else if (parameter.gainDb < 0)
        {
            coefficients.b0 = static_cast<T>((1 + root2 * k + k * k) / (1 + root2 * std::sqrt(v0) * k + v0 * k * k));
            coefficients.b1 = static_cast<T>((2 * (k * k - 1)) / (1 + root2 * std::sqrt(v0) * k + v0 * k * k));
            coefficients.b2 = static_cast<T>((1 - root2 * k + k * k) / (1 + root2 * std::sqrt(v0) * k + v0 * k * k));
            coefficients.a1 = static_cast<T>((2 * (v0 * k * k - 1)) / (1 + root2 * std::sqrt(v0) * k + v0 * k * k));
            coefficients.a2 = static_cast<T>((1 - root2 * std::sqrt(v0) * k + v0 * k * k) /
                (1 + root2 * std::sqrt(v0) * k + v0 * k * k));
        }
        else
        {
            coefficients.b0 = static_cast<T>(v0);
            coefficients.b1 = 0;
            coefficients.b2 = 0;
            coefficients.a1 = 0;
            coefficients.a2 = 0;
        }
    }

    template<class T>
    inline void ParametricEqDesigner<T>::designHighShelvingFilter(BiquadCoefficients<T>& coefficients,
        const ParametricEqParameters& parameter)
    {
        double k = std::tan((M_PI * parameter.cutoffFrequency) / m_sampleFrequency);
        double v0 = std::pow(10.0, parameter.gainDb / 20.0);
        double root2 = 1.0 / parameter.Q;

        if (v0 < 1)
        {
            v0 = 1.0 / v0;
        }

        if (parameter.gainDb > 0)
        {
            coefficients.b0 = static_cast<T>((v0 + root2 * std::sqrt(v0) * k + k * k) / (1 + root2 * k + k * k));
            coefficients.b1 = static_cast<T>((2 * (k * k - v0)) / (1 + root2 * k + k * k));
            coefficients.b2 = static_cast<T>((v0 - root2 * std::sqrt(v0) * k + k * k) / (1 + root2 * k + k * k));
            coefficients.a1 = static_cast<T>((2 * (k * k - 1)) / (1 + root2 * k + k * k));
            coefficients.a2 = static_cast<T>((1 - root2 * k + k * k) / (1 + root2 * k + k * k));
        }
        else if (parameter.gainDb < 0)
        {
            coefficients.b0 = static_cast<T>((1 + root2 * k + k * k) / (v0 + root2 * std::sqrt(v0) * k + k * k));
            coefficients.b1 = static_cast<T>((2 * (k * k - 1)) / (v0 + root2 * std::sqrt(v0) * k + k * k));
            coefficients.b2 = static_cast<T>((1 - root2 * k + k * k) / (v0 + root2 * std::sqrt(v0) * k + k * k));
            coefficients.a1 = static_cast<T>((2 * ((k * k) / v0 - 1)) / (1 + root2 / std::sqrt(v0) * k + (k * k) / v0));
            coefficients.a2 = static_cast<T>((1 - root2 / std::sqrt(v0) * k + (k * k) / v0) /
                (1 + root2 / std::sqrt(v0) * k + (k * k) / v0));
        }
        else
        {
            coefficients.b0 = static_cast<T>(v0);
            coefficients.b1 = 0;
            coefficients.b2 = 0;
            coefficients.a1 = 0;
            coefficients.a2 = 0;
        }
    }

    template<class T>
    inline void ParametricEqDesigner<T>::designPeakFilter(BiquadCoefficients<T>& coefficients,
        const ParametricEqParameters& parameter)
    {
        double w_c = (2 * M_PI * parameter.cutoffFrequency / m_sampleFrequency);
        double mu = std::pow(10.0, parameter.gainDb / 20.0);
        double k_q = 4 / (1 + mu) * std::tan(w_c / (2 * parameter.Q));
        double C_pk = (1 + k_q * mu) / (1 + k_q);

        coefficients.b0 = static_cast<T>(C_pk);
        coefficients.b1 = static_cast<T>(C_pk * (-2 * cos(w_c) / (1 + k_q * mu)));
        coefficients.b2 = static_cast<T>(C_pk * (1 - k_q * mu) / (1 + k_q * mu));

        coefficients.a1 = static_cast<T>(-2 * cos(w_c) / (1 + k_q));
        coefficients.a2 = static_cast<T>((1 - k_q) / (1 + k_q));
    }
}

#endif
