#ifndef SIGNAL_PROCESSING_FILTERS_DESIGN_GRAPHIC_EQ_DESIGNER_H
#define SIGNAL_PROCESSING_FILTERS_DESIGN_GRAPHIC_EQ_DESIGNER_H

#include <SignalProcessing/Filters/BiquadCoefficients.h>
#include <SignalProcessing/Filters/Design/Math.h>

#include <Utils/Exception/InvalidValueException.h>

#include <armadillo>

#include <vector>
#include <cmath>

namespace adaptone
{
    /*
     * Reference :
     * Rämö, Jussi & Välimäki, Vesa & Bank, Balazs. (2014).
     * High-Precision Parallel Graphic Equalizer.
     * Audio, Speech, and Language Processing,
     * IEEE/ACM Transactions on. 22. 1894-1904. 10.1109/TASLP.2014.2354241.
     */
    template<class T>
    class GraphicEqDesigner
    {
        static constexpr std::size_t MinPhaseN = 8193; // Vector size for calculating the min phase target

        double m_sampleFrequency;
        std::vector<BiquadCoefficients<T>> m_biquadCoefficients;
        T m_d0;

        arma::vec m_centerW;
        arma::vec m_gains;

        arma::vec m_optimizationW;
        arma::vec m_interpolatedGains;

        arma::cx_mat m_M;
        arma::cx_mat m_weightedM;
        arma::mat m_Mr;

        arma::cx_vec m_ht;
        arma::vec m_htr;

        arma::vec m_weight;
        arma::vec m_B;

        // Vectors for calculating ht
        arma::vec m_minPhaseW;
        arma::vec m_minPhaseGains;
        arma::vec m_minPhaseGains1Period;
        arma::vec m_minPhaseGains2Period;
        arma::cx_vec m_analyticSignal;
        arma::vec m_phaseUpsampled;
        arma::vec m_phase;

    public:
        GraphicEqDesigner(std::size_t sampleFrequency, const std::vector<double>& centerFrequencies);
        virtual ~GraphicEqDesigner();

        void update(const std::vector<double>& gainsDb);
        const std::vector<BiquadCoefficients<T>>& biquadCoefficients() const;
        T d0();

    private:
        arma::vec getPolesW();
        arma::vec getBandwidth(const arma::vec& polesW);

        void initPoles();
        void initM();

        void updateHt();
        void applyWeighting();

        void updateHtr();
        void updateMr();

        void updateBCoefficients();
    };

    template<class T>
    inline
    GraphicEqDesigner<T>::GraphicEqDesigner(std::size_t sampleFrequency, const std::vector<double>& centerFrequencies) :
        m_sampleFrequency(sampleFrequency),
        m_biquadCoefficients(centerFrequencies.size() * 2),

        m_centerW(centerFrequencies),
        m_gains(centerFrequencies.size()),

        m_optimizationW(4 * centerFrequencies.size()),
        m_interpolatedGains(m_optimizationW.size()),


        m_M(m_optimizationW.n_elem, m_biquadCoefficients.size() * 2 + 1),
        m_weightedM(arma::size(m_M)),
        m_Mr(m_M.n_rows * 2, m_M.n_cols),
        m_ht(m_optimizationW.n_elem),
        m_htr(m_ht.n_elem * 2),
        m_weight(m_ht.n_elem),
        m_B(m_M.n_cols),

        m_minPhaseW(MinPhaseN),
        m_minPhaseGains(m_minPhaseW.n_elem),
        m_minPhaseGains1Period(2 * (MinPhaseN - 1)),
        m_minPhaseGains2Period(2 * m_minPhaseGains1Period.n_elem),
        m_analyticSignal(m_minPhaseGains2Period.n_elem),
        m_phaseUpsampled(MinPhaseN),
        m_phase(m_optimizationW.n_elem)
    {
        if (!arma::vec(centerFrequencies).is_sorted() || centerFrequencies.size() < 1)
        {
            THROW_INVALID_VALUE_EXCEPTION("centerFrequencies", "");
        }

        m_centerW *= 2 * M_PI / m_sampleFrequency;

        double minFrequency = centerFrequencies[0] / 2;
        double maxFrequency = m_sampleFrequency / 2;

        m_optimizationW = arma::logspace(std::log10(minFrequency), std::log10(maxFrequency), m_optimizationW.n_elem);
        m_optimizationW *= 2 * M_PI / m_sampleFrequency;

        m_minPhaseW = arma::linspace(0, M_PI, m_minPhaseW.n_elem);

        initPoles();
        initM();

        update(arma::conv_to<std::vector<double>>::from(arma::zeros(m_centerW.n_elem)));
    }

    template<class T>
    inline GraphicEqDesigner<T>::~GraphicEqDesigner()
    {
    }

    template<class T>
    void GraphicEqDesigner<T>::update(const std::vector<double>& gainsDb)
    {
        if (m_centerW.n_elem != gainsDb.size())
        {
            THROW_INVALID_VALUE_EXCEPTION("gainsDb.size()", "");
        }

        for (std::size_t i = 0; i < m_gains.n_elem; i++)
        {
            m_gains(i) = std::pow(10, gainsDb[i] / 20.0);
        }

        updateHt();
        applyWeighting();

        updateHtr();
        updateMr();

        updateBCoefficients();
    }

    template<class T>
    inline const std::vector<BiquadCoefficients<T>>& GraphicEqDesigner<T>::biquadCoefficients() const
    {
        return m_biquadCoefficients;
    }

    template<class T>
    inline T GraphicEqDesigner<T>::d0()
    {
        return m_d0;
    }

    template<class T>
    inline arma::vec GraphicEqDesigner<T>::getPolesW()
    {
        arma::vec polesW(m_centerW.n_elem * 2);
        polesW(0) = m_centerW(0) / 2;
        polesW(1) = m_centerW(0);

        for (std::size_t i = 1; i < m_centerW.n_elem; i++)
        {
            polesW(2 * i) = (m_centerW(i - 1) + m_centerW(i)) / 2;
            polesW(2 * i + 1) = m_centerW(i);
        }

        return polesW;
    }

    template<class T>
    inline arma::vec GraphicEqDesigner<T>::getBandwidth(const arma::vec& frequencies)
    {
        arma::vec bandwidth(frequencies.n_elem);

        bandwidth(0) = frequencies(1) - frequencies(0);

        for (std::size_t i = 1; i < bandwidth.n_elem - 1; i++)
        {
            bandwidth(i) = (frequencies(i + 1) - frequencies(i - 1)) / 2;
        }

        bandwidth(bandwidth.n_elem - 1) = frequencies(bandwidth.n_elem - 1) - frequencies(bandwidth.n_elem - 2);

        return bandwidth;
    }

    template<class T>
    inline void GraphicEqDesigner<T>::initPoles()
    {
        arma::vec polesW = getPolesW();
        arma::vec bandwidth = getBandwidth(polesW);

        std::complex<double> j(0, 1);
        arma::cx_vec poles = arma::exp(-bandwidth / 2) % arma::exp(j * polesW);

        for (std::size_t i = 0; i < m_biquadCoefficients.size(); i++)
        {
            m_biquadCoefficients[i].a1 = static_cast<T>(-(poles(i) + std::conj(poles(i))).real());
            m_biquadCoefficients[i].a2 = static_cast<T>(std::abs(poles(i)) * std::abs(poles(i)));
        }
    }

    template<class T>
    inline void GraphicEqDesigner<T>::initM()
    {
        std::complex<double> j(0, 1);
        m_M.ones();

        arma::cx_vec exp1 = arma::exp(-j * m_optimizationW);
        arma::cx_vec exp2 = arma::exp(-2.0 * j * m_optimizationW);

        for (std::size_t i = 0; i < m_biquadCoefficients.size(); i++)
        {
            m_M.col(2 * i) = 1 / (1 + m_biquadCoefficients[i].a1 * exp1 + m_biquadCoefficients[i].a2 * exp2);
            m_M.col(2 * i + 1) = exp1 / (1 + m_biquadCoefficients[i].a1 * exp1 + m_biquadCoefficients[i].a2 * exp2);
        }
    }

    template<class T>
    inline void GraphicEqDesigner<T>::updateHt()
    {
        interpolateWithNaNRemoval(m_centerW, m_gains, m_optimizationW, m_interpolatedGains);
        interpolateWithNaNRemoval(m_optimizationW, m_interpolatedGains, m_minPhaseW, m_minPhaseGains);

        std::size_t N = m_minPhaseGains.n_elem - 1;

        // Calculate the periodic gain target
        m_minPhaseGains1Period(arma::span(0, N - 1)) = arma::reverse(m_minPhaseGains(arma::span(1, N)));
        m_minPhaseGains1Period(arma::span(N, m_minPhaseGains1Period.n_elem - 1)) =
            m_minPhaseGains(arma::span(0, N - 1));
        m_minPhaseGains1Period = arma::log(m_minPhaseGains1Period);

        m_minPhaseGains2Period(arma::span(0, m_minPhaseGains1Period.n_elem - 1)) = m_minPhaseGains1Period;
        m_minPhaseGains2Period(arma::span(m_minPhaseGains1Period.n_elem, m_minPhaseGains2Period.n_elem - 1)) =
            m_minPhaseGains1Period;

        // Calculate the min phase target
        hilbert(m_minPhaseGains2Period, m_analyticSignal);
        m_phaseUpsampled = arma::imag(m_analyticSignal(arma::span(N, 2 * N)));
        interpolateWithNaNRemoval(m_minPhaseW, m_phaseUpsampled, m_optimizationW, m_phase);

        std::complex<double> j(0, 1);
        m_ht = m_interpolatedGains % arma::exp(-j * m_phase);
    }

    template<class T>
    inline void GraphicEqDesigner<T>::applyWeighting()
    {
        m_weight = arma::abs(m_ht);
        m_weight = arma::sqrt(1 / (m_weight % m_weight));

        m_weightedM = m_M;
        for (std::size_t i = 0; i < m_weight.n_elem; i++)
        {
            m_weightedM.row(i) *= m_weight(i);
            m_ht(i) *= m_weight(i);
        }
    }

    template<class T>
    inline void GraphicEqDesigner<T>::updateHtr()
    {
        m_htr(arma::span(0, m_ht.n_elem - 1)) = arma::real(m_ht);
        m_htr(arma::span(m_ht.n_elem, m_htr.n_elem - 1)) = arma::imag(m_ht);
    }

    template<class T>
    inline void GraphicEqDesigner<T>::updateMr()
    {
        m_Mr.rows(arma::span(0, m_weightedM.n_rows - 1)) = arma::real(m_weightedM);
        m_Mr.rows(arma::span(m_weightedM.n_rows, m_Mr.n_rows - 1)) = arma::imag(m_weightedM);
    }

    template<class T>
    inline void GraphicEqDesigner<T>::updateBCoefficients()
    {
        if (arma::solve(m_B, m_Mr, m_htr))
        {
            for (std::size_t i = 0; i < m_biquadCoefficients.size(); i++)
            {
                m_biquadCoefficients[i].b0 = static_cast<T>(m_B(2 * i));
                m_biquadCoefficients[i].b1 = static_cast<T>(m_B(2 * i + 1));
            }

            m_d0 = m_B(m_B.n_elem - 1);
        }
    }
}

#endif
