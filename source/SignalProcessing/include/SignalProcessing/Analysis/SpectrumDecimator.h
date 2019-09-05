#ifndef SIGNAL_PROCESSING_ANALYSIS_REALTIME_SPECTRUM_ANALYSER_H
#define SIGNAL_PROCESSING_ANALYSIS_REALTIME_SPECTRUM_ANALYSER_H

#include <Utils/Data/SpectrumPoint.h>

#include <armadillo>

#include <vector>

namespace adaptone
{
    class SpectrumDecimatorBucket
    {
        double m_frequency;
        std::size_t m_lowerBoundIndex;
        std::size_t m_upperBoundIndex;

    public:
        SpectrumDecimatorBucket(double frequency, std::size_t lowerBoundIndex, std::size_t upperBoundIndex);
        virtual ~SpectrumDecimatorBucket();

        double frequency() const;
        std::size_t lowerBoundIndex() const;
        std::size_t upperBoundIndex() const;
    };

    inline double SpectrumDecimatorBucket::frequency() const
    {
        return m_frequency;
    }

    inline std::size_t SpectrumDecimatorBucket::lowerBoundIndex() const
    {
        return m_lowerBoundIndex;
    }

    inline std::size_t SpectrumDecimatorBucket::upperBoundIndex() const
    {
        return m_upperBoundIndex;
    }

    class SpectrumDecimator
    {
        std::vector<SpectrumDecimatorBucket> m_buckets;
        std::size_t m_positiveFrequencySpectrumSize;
        double m_nyquistFrequency;

    public:
        SpectrumDecimator(std::size_t positiveFrequencySpectrumSize, std::size_t sampleFrequency,
            std::size_t pointCountPerDecade);
        virtual ~SpectrumDecimator();

        const std::vector<SpectrumDecimatorBucket>& buckets() const;

        std::vector<SpectrumPoint> getDecimatedAmplitudes(const arma::fvec& positiveFrequencyAmplitudes);

    private:
        void createBucketsFromIntersectionIndexes(const arma::vec& intersectionIndexes);
    };

    inline const std::vector<SpectrumDecimatorBucket>& SpectrumDecimator::buckets() const
    {
        return m_buckets;
    }
}

#endif
