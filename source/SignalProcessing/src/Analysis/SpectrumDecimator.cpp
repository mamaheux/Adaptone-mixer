#include <SignalProcessing/Analysis/SpectrumDecimator.h>

#include <Utils/Exception/InvalidValueException.h>

#include <cmath>

using namespace adaptone;
using namespace std;

SpectrumDecimatorBucket::SpectrumDecimatorBucket(double frequency, size_t lowerBoundIndex, size_t upperBoundIndex) :
    m_frequency(frequency), m_lowerBoundIndex(lowerBoundIndex), m_upperBoundIndex(upperBoundIndex)
{
}

SpectrumDecimatorBucket::~SpectrumDecimatorBucket()
{
}

SpectrumDecimator::SpectrumDecimator(size_t positiveFrequencySpectrumSize, size_t sampleFrequency, size_t pointCountPerDecade) :
    m_positiveFrequencySpectrumSize(positiveFrequencySpectrumSize),
    m_nyquistFrequency(static_cast<double>(sampleFrequency) / 2)
{
    double minimumFrequency = m_nyquistFrequency / positiveFrequencySpectrumSize;
    size_t pointCount = static_cast<size_t>(log10(m_nyquistFrequency) * pointCountPerDecade);

    arma::vec intersectionFrequencies = arma::logspace(log10(minimumFrequency),
        log10(m_nyquistFrequency),
        pointCount + 1);
    arma::vec intersectionIndexes =
        arma::floor(intersectionFrequencies * positiveFrequencySpectrumSize / m_nyquistFrequency);

    createBucketsFromIntersectionIndexes(intersectionIndexes);
}

SpectrumDecimator::~SpectrumDecimator()
{
}

vector<SpectrumPoint> SpectrumDecimator::getDecimatedAmplitudes(const arma::fvec& positiveFrequencyAmplitudes)
{
    if (positiveFrequencyAmplitudes.n_elem != m_positiveFrequencySpectrumSize)
    {
        THROW_INVALID_VALUE_EXCEPTION("The spectrum has the wrong size.", "");
    }

    vector<SpectrumPoint> decimatedAmplitude;
    decimatedAmplitude.reserve(m_buckets.size());

    for (SpectrumDecimatorBucket& bucket : m_buckets)
    {
        double amplitude =
            arma::mean(positiveFrequencyAmplitudes(arma::span(bucket.lowerBoundIndex(), bucket.upperBoundIndex())));
        decimatedAmplitude.emplace_back(bucket.frequency(), amplitude);
    }

    return decimatedAmplitude;
}

void SpectrumDecimator::createBucketsFromIntersectionIndexes(const arma::vec& intersectionIndexes)
{
    double frequencyStep = m_nyquistFrequency / m_positiveFrequencySpectrumSize;
    for (size_t i = 0; i < intersectionIndexes.n_elem - 1; i++)
    {
        size_t lowerBoundIndex = static_cast<size_t>(intersectionIndexes(i));
        size_t upperBoundIndex = static_cast<size_t>(intersectionIndexes(i + 1)) - 1;

        if (lowerBoundIndex > upperBoundIndex)
        {
            continue;
        }

        double frequency = static_cast<double>(lowerBoundIndex + upperBoundIndex) / 2 * frequencyStep;
        m_buckets.emplace_back(frequency, lowerBoundIndex, upperBoundIndex);
    }
}
