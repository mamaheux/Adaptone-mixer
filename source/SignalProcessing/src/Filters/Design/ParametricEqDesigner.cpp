#include <SignalProcessing/Filters/Design/ParametricEqDesigner.h>

#include <Utils/Exception/InvalidValueException.h>

#include <armadillo>

#include <cmath>

using namespace adaptone;
using namespace std;

class ArmaParametricEqDesignerPrivate : public ParametricEqDesignerPrivate
{
    size_t m_filterCount;
    double m_sampleFrequency;

    vector<BiquadCoefficients<float>> m_floatBiquadCoefficients;
    vector<BiquadCoefficients<double>> m_doubleBiquadCoefficients;

public:
    ArmaParametricEqDesignerPrivate(size_t filterCount, size_t sampleFrequency);
    ~ArmaParametricEqDesignerPrivate() override;

    DECLARE_NOT_COPYABLE(ArmaParametricEqDesignerPrivate);
    DECLARE_NOT_MOVABLE(ArmaParametricEqDesignerPrivate);

    void update(const vector<ParametricEqParameters>& parameters) override;
    const vector<BiquadCoefficients<float>>& floatBiquadCoefficients() const override;
    const vector<BiquadCoefficients<double>>& doubleBiquadCoefficients() const override;
    vector<double> gainsDb(const vector<double>& frequencies) const override;

private:
    void designLowShelvingFilter(BiquadCoefficients<double>& coefficients, const ParametricEqParameters& parameter);
    void designHighShelvingFilter(BiquadCoefficients<double>& coefficients, const ParametricEqParameters& parameter);
    void designPeakFilter(BiquadCoefficients<double>& coefficients, const ParametricEqParameters& parameter);
};

ArmaParametricEqDesignerPrivate::ArmaParametricEqDesignerPrivate(size_t filterCount, size_t sampleFrequency) :
    m_filterCount(filterCount), m_sampleFrequency(sampleFrequency), m_floatBiquadCoefficients(filterCount),
    m_doubleBiquadCoefficients(filterCount)
{
    if (m_filterCount < 2)
    {
        THROW_INVALID_VALUE_EXCEPTION("filterCount", "");
    }
}

ArmaParametricEqDesignerPrivate::~ArmaParametricEqDesignerPrivate()
{
}

void ArmaParametricEqDesignerPrivate::update(const vector<ParametricEqParameters>& parameters)
{
    if (m_filterCount != parameters.size())
    {
        THROW_INVALID_VALUE_EXCEPTION("parameters.size()", "");
    }

    designLowShelvingFilter(m_doubleBiquadCoefficients[0], parameters[0]);
    designHighShelvingFilter(m_doubleBiquadCoefficients[m_filterCount - 1], parameters[m_filterCount - 1]);

    for (size_t i = 1; i < m_filterCount - 1; i++)
    {
        designPeakFilter(m_doubleBiquadCoefficients[i], parameters[i]);
    }

    for (size_t i = 0; i < m_filterCount; i++)
    {
        m_floatBiquadCoefficients[i].b0 = static_cast<float>(m_doubleBiquadCoefficients[i].b0);
        m_floatBiquadCoefficients[i].b1 = static_cast<float>(m_doubleBiquadCoefficients[i].b1);
        m_floatBiquadCoefficients[i].b2 = static_cast<float>(m_doubleBiquadCoefficients[i].b2);
        m_floatBiquadCoefficients[i].a1 = static_cast<float>(m_doubleBiquadCoefficients[i].a1);
        m_floatBiquadCoefficients[i].a2 = static_cast<float>(m_doubleBiquadCoefficients[i].a2);
    }
}

const vector<BiquadCoefficients<float>>& ArmaParametricEqDesignerPrivate::floatBiquadCoefficients() const
{
    return m_floatBiquadCoefficients;
}

const vector<BiquadCoefficients<double>>& ArmaParametricEqDesignerPrivate::doubleBiquadCoefficients() const
{
    return m_doubleBiquadCoefficients;
}

vector<double> ArmaParametricEqDesignerPrivate::gainsDb(const vector<double>& frequencies) const
{
    arma::vec w(frequencies);
    w = 2 * M_PI * w / m_sampleFrequency;

    complex<double> j(0, 1);
    arma::cx_vec jw = -j * w;
    arma::cx_vec jw2 = 2 * jw;

    arma::cx_vec h(frequencies.size());
    h.ones();

    for (const BiquadCoefficients<double>& c : m_doubleBiquadCoefficients)
    {
        h %= (c.b0 + c.b1 * arma::exp(jw) + c.b2 * arma::exp(jw2)) /
            (1 + c.a1 * arma::exp(jw) + c.a2 * arma::exp(jw2));
    }

    return arma::conv_to<vector<double>>::from(20 * arma::log10(arma::abs(h)));
}

void ArmaParametricEqDesignerPrivate::designLowShelvingFilter(BiquadCoefficients<double>& coefficients,
    const ParametricEqParameters& parameter)
{
    double k = tan((M_PI * parameter.cutoffFrequency) / m_sampleFrequency);
    double v0 = pow(10.0, parameter.gainDb / 20.0);
    double root2 = 1.0 / parameter.Q;

    if (v0 < 1)
    {
        v0 = 1.0 / v0;
    }

    if (parameter.gainDb > 0)
    {
        coefficients.b0 = (1 + sqrt(v0) * root2 * k + v0 * k * k) / (1 + root2 * k + k * k);
        coefficients.b1 = (2 * (v0 * k * k - 1)) / (1 + root2 * k + k * k);
        coefficients.b2 = (1 - sqrt(v0) * root2 * k + v0 * k * k) / (1 + root2 * k + k * k);
        coefficients.a1 = (2 * (k * k - 1)) / (1 + root2 * k + k * k);
        coefficients.a2 = (1 - root2 * k + k * k) / (1 + root2 * k + k * k);
    }
    else if (parameter.gainDb < 0)
    {
        coefficients.b0 = (1 + root2 * k + k * k) / (1 + root2 * sqrt(v0) * k + v0 * k * k);
        coefficients.b1 = (2 * (k * k - 1)) / (1 + root2 * sqrt(v0) * k + v0 * k * k);
        coefficients.b2 = (1 - root2 * k + k * k) / (1 + root2 * sqrt(v0) * k + v0 * k * k);
        coefficients.a1 = (2 * (v0 * k * k - 1)) / (1 + root2 * sqrt(v0) * k + v0 * k * k);
        coefficients.a2 = (1 - root2 * sqrt(v0) * k + v0 * k * k) / (1 + root2 * sqrt(v0) * k + v0 * k * k);
    }
    else
    {
        coefficients.b0 = v0;
        coefficients.b1 = 0;
        coefficients.b2 = 0;
        coefficients.a1 = 0;
        coefficients.a2 = 0;
    }
}

void ArmaParametricEqDesignerPrivate::designHighShelvingFilter(BiquadCoefficients<double>& coefficients,
    const ParametricEqParameters& parameter)
{
    double k = tan((M_PI * parameter.cutoffFrequency) / m_sampleFrequency);
    double v0 = pow(10.0, parameter.gainDb / 20.0);
    double root2 = 1.0 / parameter.Q;

    if (v0 < 1)
    {
        v0 = 1.0 / v0;
    }

    if (parameter.gainDb > 0)
    {
        coefficients.b0 = (v0 + root2 * sqrt(v0) * k + k * k) / (1 + root2 * k + k * k);
        coefficients.b1 = (2 * (k * k - v0)) / (1 + root2 * k + k * k);
        coefficients.b2 = (v0 - root2 * sqrt(v0) * k + k * k) / (1 + root2 * k + k * k);
        coefficients.a1 = (2 * (k * k - 1)) / (1 + root2 * k + k * k);
        coefficients.a2 = (1 - root2 * k + k * k) / (1 + root2 * k + k * k);
    }
    else if (parameter.gainDb < 0)
    {
        coefficients.b0 = (1 + root2 * k + k * k) / (v0 + root2 * sqrt(v0) * k + k * k);
        coefficients.b1 = (2 * (k * k - 1)) / (v0 + root2 * sqrt(v0) * k + k * k);
        coefficients.b2 = (1 - root2 * k + k * k) / (v0 + root2 * sqrt(v0) * k + k * k);
        coefficients.a1 = (2 * ((k * k) / v0 - 1)) / (1 + root2 / sqrt(v0) * k + (k * k) / v0);
        coefficients.a2 = (1 - root2 / sqrt(v0) * k + (k * k) / v0) / (1 + root2 / sqrt(v0) * k + (k * k) / v0);
    }
    else
    {
        coefficients.b0 = v0;
        coefficients.b1 = 0;
        coefficients.b2 = 0;
        coefficients.a1 = 0;
        coefficients.a2 = 0;
    }
}

void ArmaParametricEqDesignerPrivate::designPeakFilter(BiquadCoefficients<double>& coefficients,
    const ParametricEqParameters& parameter)
{
    double w_c = (2 * M_PI * parameter.cutoffFrequency / m_sampleFrequency);
    double mu = pow(10.0, parameter.gainDb / 20.0);
    double k_q = 4 / (1 + mu) * tan(w_c / (2 * parameter.Q));
    double C_pk = (1 + k_q * mu) / (1 + k_q);

    coefficients.b0 = C_pk;
    coefficients.b1 = C_pk * (-2 * cos(w_c) / (1 + k_q * mu));
    coefficients.b2 = C_pk * (1 - k_q * mu) / (1 + k_q * mu);

    coefficients.a1 = -2 * cos(w_c) / (1 + k_q);
    coefficients.a2 = (1 - k_q) / (1 + k_q);
}

unique_ptr<ParametricEqDesignerPrivate> adaptone::createParametricEqDesignerPrivate(size_t filterCount,
    size_t sampleFrequency)
{
    return make_unique<ArmaParametricEqDesignerPrivate>(filterCount, sampleFrequency);
}
