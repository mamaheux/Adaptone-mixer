#ifndef UTILS_DATA_SPECTRUM_POINT_H
#define UTILS_DATA_SPECTRUM_POINT_H

namespace adaptone
{
    class SpectrumPoint
    {
        double m_frequency;
        double m_amplitude;

    public:
        SpectrumPoint();
        SpectrumPoint(double frequency, double amplitude);
        virtual ~SpectrumPoint();

        double frequency() const;
        double amplitude() const;
    };

    inline double SpectrumPoint::frequency() const
    {
        return m_frequency;
    }

    inline double SpectrumPoint::amplitude() const
    {
        return m_amplitude;
    }
}

#endif

