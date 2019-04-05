#ifndef SIGNAL_PROCESSING_FILTERS_DESIGN_PARAMETRIC_EQ_DESIGNER_H
#define SIGNAL_PROCESSING_FILTERS_DESIGN_PARAMETRIC_EQ_DESIGNER_H

#include <SignalProcessing/Filters/BiquadCoefficients.h>
#include <SignalProcessing/Filters/ParametricEqParameters.h>

#include <Utils/ClassMacro.h>
#include <Utils/TypeTraits.h>

#include <memory>
#include <vector>

namespace adaptone
{
    class ParametricEqDesignerPrivate
    {
    public:
        ParametricEqDesignerPrivate();
        virtual ~ParametricEqDesignerPrivate();

        DECLARE_NOT_COPYABLE(ParametricEqDesignerPrivate);
        DECLARE_NOT_MOVABLE(ParametricEqDesignerPrivate);

        virtual void update(const std::vector<ParametricEqParameters>& parameters) = 0;
        virtual const std::vector<BiquadCoefficients<float>>& floatBiquadCoefficients() const = 0;
        virtual const std::vector<BiquadCoefficients<double>>& doubleBiquadCoefficients() const = 0;
        virtual std::vector<double> gainsDb(const std::vector<double>& frequencies) const = 0;
    };

    inline ParametricEqDesignerPrivate::ParametricEqDesignerPrivate()
    {
    }

    inline ParametricEqDesignerPrivate::~ParametricEqDesignerPrivate()
    {
    }

    std::unique_ptr<ParametricEqDesignerPrivate> createParametricEqDesignerPrivate(std::size_t filterCount,
        std::size_t sampleFrequency);

    template<class T>
    class ParametricEqDesigner
    {
    protected:
        std::unique_ptr<ParametricEqDesignerPrivate> m_parametricEqDesignerPrivate;

    public:
        ParametricEqDesigner(std::size_t filterCount, std::size_t sampleFrequency);
        virtual ~ParametricEqDesigner();

        DECLARE_NOT_COPYABLE(ParametricEqDesigner);
        DECLARE_NOT_MOVABLE(ParametricEqDesigner);

        void update(const std::vector<ParametricEqParameters>& parameters);
        const std::vector<BiquadCoefficients<T>>& biquadCoefficients() const;
        std::vector<double> gainsDb(const std::vector<double>& frequencies) const;
    };

    template<class T>
    inline ParametricEqDesigner<T>::ParametricEqDesigner(std::size_t filterCount,
        std::size_t sampleFrequency)
    {
        m_parametricEqDesignerPrivate = move(createParametricEqDesignerPrivate(filterCount, sampleFrequency));
    }

    template<class T>
    inline ParametricEqDesigner<T>::~ParametricEqDesigner()
    {
    }

    template<class T>
    inline void ParametricEqDesigner<T>::update(const std::vector<ParametricEqParameters>& parameters)
    {
        m_parametricEqDesignerPrivate->update(parameters);
    }

    template<class T>
    inline std::vector<double> ParametricEqDesigner<T>::gainsDb(const std::vector<double>& frequencies) const
    {
        return m_parametricEqDesignerPrivate->gainsDb(frequencies);
    }

    template<class T>
    inline const std::vector<BiquadCoefficients<T>>& ParametricEqDesigner<T>::biquadCoefficients() const
    {
        static_assert(AlwaysFalse<T>::value, "Not supported type");
    }

    template<>
    inline const std::vector<BiquadCoefficients<float>>& ParametricEqDesigner<float>::biquadCoefficients() const
    {
        return m_parametricEqDesignerPrivate->floatBiquadCoefficients();
    }

    template<>
    inline const std::vector<BiquadCoefficients<double>>& ParametricEqDesigner<double>::biquadCoefficients() const
    {
        return m_parametricEqDesignerPrivate->doubleBiquadCoefficients();
    }
}

#endif
