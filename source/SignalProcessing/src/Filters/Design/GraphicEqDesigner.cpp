#include <SignalProcessing/Filters/Design/GraphicEqDesigner.h>
#include <SignalProcessing/Utils/Math.h>

#include <Utils/Exception/InvalidValueException.h>

#include <armadillo>

using namespace adaptone;
using namespace std;

class ArmaGraphicEqDesignerPrivate : public GraphicEqDesignerPrivate
{
    static constexpr size_t MinPhaseN = 8193; // Vector size for calculating the min phase target

    double m_sampleFrequency;
    vector<BiquadCoefficients<float>> m_floatBiquadCoefficients;
    vector<BiquadCoefficients<double>> m_doubleBiquadCoefficients;
    double m_d0;

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
    arma::cx_vec m_minPhaseGains2Period;
    arma::cx_vec m_analyticSignal;
    arma::vec m_phaseUpsampled;
    arma::vec m_phase;

public:
    ArmaGraphicEqDesignerPrivate(size_t sampleFrequency, const vector<double>& centerFrequencies);
    ~ArmaGraphicEqDesignerPrivate() override;

    DECLARE_NOT_COPYABLE(ArmaGraphicEqDesignerPrivate);
    DECLARE_NOT_MOVABLE(ArmaGraphicEqDesignerPrivate);

    void update(const vector<double>& gains) override;
    void update(const vector<BiquadCoefficients<float>>& biquadCoefficients, float d0) override;
    void update(const vector<BiquadCoefficients<double>>& biquadCoefficients, double d0) override;

    const vector<BiquadCoefficients<float>>& floatBiquadCoefficients() const override;
    const vector<BiquadCoefficients<double>>& doubleBiquadCoefficients() const override;
    double d0() const override;

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

ArmaGraphicEqDesignerPrivate::ArmaGraphicEqDesignerPrivate(size_t sampleFrequency,
    const vector<double>& centerFrequencies) :
    m_sampleFrequency(sampleFrequency),
    m_floatBiquadCoefficients(centerFrequencies.size() * 2),
    m_doubleBiquadCoefficients(m_floatBiquadCoefficients.size()),

    m_centerW(centerFrequencies),
    m_gains(centerFrequencies.size()),

    m_optimizationW(4 * centerFrequencies.size()),
    m_interpolatedGains(m_optimizationW.size()),


    m_M(m_optimizationW.n_elem, m_floatBiquadCoefficients.size() * 2 + 1),
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

    m_optimizationW = arma::logspace(log10(minFrequency), log10(maxFrequency), m_optimizationW.n_elem);
    m_optimizationW *= 2 * M_PI / m_sampleFrequency;

    m_minPhaseW = arma::linspace(0, M_PI, m_minPhaseW.n_elem);

    initPoles();
    initM();

    update(arma::conv_to<vector<double>>::from(arma::ones(m_centerW.n_elem)));
}

ArmaGraphicEqDesignerPrivate::~ArmaGraphicEqDesignerPrivate()
{
}

void ArmaGraphicEqDesignerPrivate::update(const vector<double>& gains)
{
    if (m_centerW.n_elem != gains.size())
    {
        THROW_INVALID_VALUE_EXCEPTION("gains.size()", "");
    }
    m_gains = gains;

    updateHt();
    applyWeighting();

    updateHtr();
    updateMr();

    updateBCoefficients();
}

void ArmaGraphicEqDesignerPrivate::update(const vector<BiquadCoefficients<float>>& biquadCoefficients, float d0)
{
    if (m_floatBiquadCoefficients.size() != biquadCoefficients.size())
    {
        THROW_INVALID_VALUE_EXCEPTION("biquadCoefficients.size()", "");
    }

    m_floatBiquadCoefficients.assign(biquadCoefficients.cbegin(), biquadCoefficients.cend());

    for (size_t i = 0; i < m_doubleBiquadCoefficients.size(); i++)
    {
        m_doubleBiquadCoefficients[i].b0 = biquadCoefficients[i].b0;
        m_doubleBiquadCoefficients[i].b1 = biquadCoefficients[i].b1;
        m_doubleBiquadCoefficients[i].b2 = biquadCoefficients[i].b2;
        m_doubleBiquadCoefficients[i].a1 = biquadCoefficients[i].a1;
        m_doubleBiquadCoefficients[i].a2 = biquadCoefficients[i].a2;
    }

    m_d0 = d0;
}

void ArmaGraphicEqDesignerPrivate::update(const vector<BiquadCoefficients<double>>& biquadCoefficients, double d0)
{
    if (m_floatBiquadCoefficients.size() != biquadCoefficients.size())
    {
        THROW_INVALID_VALUE_EXCEPTION("biquadCoefficients.size()", "");
    }

    m_doubleBiquadCoefficients.assign(biquadCoefficients.cbegin(), biquadCoefficients.cend());
    for (size_t i = 0; i < m_floatBiquadCoefficients.size(); i++)
    {
        m_floatBiquadCoefficients[i].b0 = static_cast<float>(biquadCoefficients[i].b0);
        m_floatBiquadCoefficients[i].b1 = static_cast<float>(biquadCoefficients[i].b1);
        m_floatBiquadCoefficients[i].b2 = static_cast<float>(biquadCoefficients[i].b2);
        m_floatBiquadCoefficients[i].a1 = static_cast<float>(biquadCoefficients[i].a1);
        m_floatBiquadCoefficients[i].a2 = static_cast<float>(biquadCoefficients[i].a2);
    }

    m_d0 = d0;
}

const vector<BiquadCoefficients<float>>& ArmaGraphicEqDesignerPrivate::floatBiquadCoefficients() const
{
    return m_floatBiquadCoefficients;
}

const vector<BiquadCoefficients<double>>& ArmaGraphicEqDesignerPrivate::doubleBiquadCoefficients() const
{
    return m_doubleBiquadCoefficients;
}

double ArmaGraphicEqDesignerPrivate::d0() const
{
    return m_d0;
}

arma::vec ArmaGraphicEqDesignerPrivate::getPolesW()
{
    arma::vec polesW(m_centerW.n_elem * 2);
    polesW(0) = m_centerW(0) / 2;
    polesW(1) = m_centerW(0);

    for (size_t i = 1; i < m_centerW.n_elem; i++)
    {
        polesW(2 * i) = (m_centerW(i - 1) + m_centerW(i)) / 2;
        polesW(2 * i + 1) = m_centerW(i);
    }

    return polesW;
}

arma::vec ArmaGraphicEqDesignerPrivate::getBandwidth(const arma::vec& frequencies)
{
    arma::vec bandwidth(frequencies.n_elem);

    bandwidth(0) = frequencies(1) - frequencies(0);

    for (size_t i = 1; i < bandwidth.n_elem - 1; i++)
    {
        bandwidth(i) = (frequencies(i + 1) - frequencies(i - 1)) / 2;
    }

    bandwidth(bandwidth.n_elem - 1) = frequencies(bandwidth.n_elem - 1) - frequencies(bandwidth.n_elem - 2);

    return bandwidth;
}

void ArmaGraphicEqDesignerPrivate::initPoles()
{
    arma::vec polesW = getPolesW();
    arma::vec bandwidth = getBandwidth(polesW);

    complex<double> j(0, 1);
    arma::cx_vec poles = arma::exp(-bandwidth / 2) % arma::exp(j * polesW);

    for (size_t i = 0; i < m_floatBiquadCoefficients.size(); i++)
    {
        m_floatBiquadCoefficients[i].a1 = static_cast<float>(-(poles(i) + conj(poles(i))).real());
        m_floatBiquadCoefficients[i].a2 = static_cast<float>(abs(poles(i)) * abs(poles(i)));

        m_doubleBiquadCoefficients[i].a1 = -(poles(i) + conj(poles(i))).real();
        m_doubleBiquadCoefficients[i].a2 = abs(poles(i)) * abs(poles(i));
    }
}

void ArmaGraphicEqDesignerPrivate::initM()
{
    complex<double> j(0, 1);
    m_M.ones();

    arma::cx_vec exp1 = arma::exp(-j * m_optimizationW);
    arma::cx_vec exp2 = arma::exp(-2.0 * j * m_optimizationW);

    for (size_t i = 0; i < m_doubleBiquadCoefficients.size(); i++)
    {
        m_M.col(2 * i) = 1 / (1 + m_doubleBiquadCoefficients[i].a1 * exp1 + m_doubleBiquadCoefficients[i].a2 * exp2);
        m_M.col(2 * i + 1) = exp1 /
            (1 + m_doubleBiquadCoefficients[i].a1 * exp1 + m_doubleBiquadCoefficients[i].a2 * exp2);
    }
}

void ArmaGraphicEqDesignerPrivate::updateHt()
{
    interpolateWithNaNRemoval(m_centerW, m_gains, m_optimizationW, m_interpolatedGains);
    interpolateWithNaNRemoval(m_optimizationW, m_interpolatedGains, m_minPhaseW, m_minPhaseGains);

    size_t N = m_minPhaseGains.n_elem - 1;

    // Calculate the periodic gain target
    m_minPhaseGains1Period(arma::span(0, N - 1)) = arma::reverse(m_minPhaseGains(arma::span(1, N)));
    m_minPhaseGains1Period(arma::span(N, m_minPhaseGains1Period.n_elem - 1)) =
        m_minPhaseGains(arma::span(0, N - 1));
    m_minPhaseGains1Period = arma::log(m_minPhaseGains1Period);

    m_minPhaseGains2Period(arma::span(0, m_minPhaseGains1Period.n_elem - 1)) = arma::conv_to<arma::cx_vec>::from(m_minPhaseGains1Period);
    m_minPhaseGains2Period(arma::span(m_minPhaseGains1Period.n_elem, m_minPhaseGains2Period.n_elem - 1)) =
        arma::conv_to<arma::cx_vec>::from(m_minPhaseGains1Period);

    // Calculate the min phase target
    hilbert(m_minPhaseGains2Period, m_analyticSignal);
    m_phaseUpsampled = arma::imag(m_analyticSignal(arma::span(N, 2 * N)));
    interpolateWithNaNRemoval(m_minPhaseW, m_phaseUpsampled, m_optimizationW, m_phase);

    complex<double> j(0, 1);
    m_ht = m_interpolatedGains % arma::exp(-j * m_phase);
}

void ArmaGraphicEqDesignerPrivate::applyWeighting()
{
    m_weight = arma::abs(m_ht);
    m_weight = arma::sqrt(1 / (m_weight % m_weight));

    m_weightedM = m_M;
    for (size_t i = 0; i < m_weight.n_elem; i++)
    {
        m_weightedM.row(i) *= m_weight(i);
        m_ht(i) *= m_weight(i);
    }
}

void ArmaGraphicEqDesignerPrivate::updateHtr()
{
    m_htr(arma::span(0, m_ht.n_elem - 1)) = arma::real(m_ht);
    m_htr(arma::span(m_ht.n_elem, m_htr.n_elem - 1)) = arma::imag(m_ht);
}

void ArmaGraphicEqDesignerPrivate::updateMr()
{
    m_Mr.rows(arma::span(0, m_weightedM.n_rows - 1)) = arma::real(m_weightedM);
    m_Mr.rows(arma::span(m_weightedM.n_rows, m_Mr.n_rows - 1)) = arma::imag(m_weightedM);
}

void ArmaGraphicEqDesignerPrivate::updateBCoefficients()
{
    if (arma::solve(m_B, m_Mr, m_htr))
    {
        for (size_t i = 0; i < m_floatBiquadCoefficients.size(); i++)
        {
            m_floatBiquadCoefficients[i].b0 = static_cast<float>(m_B(2 * i));
            m_floatBiquadCoefficients[i].b1 = static_cast<float>(m_B(2 * i + 1));

            m_doubleBiquadCoefficients[i].b0 = m_B(2 * i);
            m_doubleBiquadCoefficients[i].b1 = m_B(2 * i + 1);
        }

        m_d0 = m_B(m_B.n_elem - 1);
    }
}

unique_ptr<GraphicEqDesignerPrivate> adaptone::createGraphicEqDesignerPrivate(size_t sampleFrequency,
    const vector<double>& centerFrequencies)
{
    return make_unique<ArmaGraphicEqDesignerPrivate>(sampleFrequency, centerFrequencies);
}
