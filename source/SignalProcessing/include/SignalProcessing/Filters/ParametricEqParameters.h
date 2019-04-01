#ifndef SIGNAL_PROCESSING_FILTERS_PARAMETRIC_EQ_PARAMETERS_H
#define SIGNAL_PROCESSING_FILTERS_PARAMETRIC_EQ_PARAMETERS_H

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

    inline ParametricEqParameters::ParametricEqParameters(double cutoffFrequency, double Q, double gainDb) :
        cutoffFrequency(cutoffFrequency), Q(Q), gainDb(gainDb)
    {
    }

    inline ParametricEqParameters::~ParametricEqParameters()
    {
    }
}

#endif
