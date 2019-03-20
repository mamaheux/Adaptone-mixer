#ifndef SIGNAL_PROCESSING_FILTERS_BIQUAD_COEFFICIENTS_H
#define SIGNAL_PROCESSING_FILTERS_BIQUAD_COEFFICIENTS_H

namespace adaptone
{
    template<class T>
    struct BiquadCoefficients
    {
        T b0;
        T b1;
        T b2;
        T a1;
        T a2;

        BiquadCoefficients();
        BiquadCoefficients(T b0,
            T b1,
            T b2,
            T a1,
            T a2);
        virtual ~BiquadCoefficients();
    };

    template<class T>
    inline BiquadCoefficients<T>::BiquadCoefficients() : b0(1), b1(0), b2(0), a1(0), a2(0)
    {
    }

    template<class T>
    inline BiquadCoefficients<T>::BiquadCoefficients(T b0,
        T b1,
        T b2,
        T a1,
        T a2) : b0(b0), b1(b1), b2(b2), a1(a1), a2(a2)
    {
    }

    template<class T>
    inline BiquadCoefficients<T>::~BiquadCoefficients()
    {
    }
}

#endif
