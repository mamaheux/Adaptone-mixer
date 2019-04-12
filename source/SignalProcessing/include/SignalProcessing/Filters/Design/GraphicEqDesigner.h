#ifndef SIGNAL_PROCESSING_FILTERS_DESIGN_GRAPHIC_EQ_DESIGNER_H
#define SIGNAL_PROCESSING_FILTERS_DESIGN_GRAPHIC_EQ_DESIGNER_H

#include <SignalProcessing/Filters/BiquadCoefficients.h>

#include <Utils/ClassMacro.h>
#include <Utils/TypeTraits.h>

#include <memory>
#include <vector>

namespace adaptone
{
    /*
     * Reference :
     * Rämö, Jussi & Välimäki, Vesa & Bank, Balazs. (2014).
     * High-Precision Parallel Graphic Equalizer.
     * Audio, Speech, and Language Processing,
     * IEEE/ACM Transactions on. 22. 1894-1904. 10.1109/TASLP.2014.2354241.
     */
    class GraphicEqDesignerPrivate
    {
    public:
        GraphicEqDesignerPrivate();
        virtual ~GraphicEqDesignerPrivate();

        DECLARE_NOT_COPYABLE(GraphicEqDesignerPrivate);
        DECLARE_NOT_MOVABLE(GraphicEqDesignerPrivate);

        virtual void update(const std::vector<double>& gainsDb) = 0;
        virtual void update(const std::vector<BiquadCoefficients<float>>& biquadCoefficients, float d0) = 0;
        virtual void update(const std::vector<BiquadCoefficients<double>>& biquadCoefficients, double d0) = 0;

        virtual const std::vector<BiquadCoefficients<float>>& floatBiquadCoefficients() const = 0;
        virtual const std::vector<BiquadCoefficients<double>>& doubleBiquadCoefficients() const = 0;
        virtual double d0() const = 0;
    };

    inline GraphicEqDesignerPrivate::GraphicEqDesignerPrivate()
    {
    }

    inline GraphicEqDesignerPrivate::~GraphicEqDesignerPrivate()
    {
    }

    std::unique_ptr<GraphicEqDesignerPrivate> createGraphicEqDesignerPrivate(std::size_t sampleFrequency,
        const std::vector<double>& centerFrequencies);

    template<class T>
    class GraphicEqDesigner
    {
        std::unique_ptr<GraphicEqDesignerPrivate> m_graphicEqDesignerPrivate;

    public:
        GraphicEqDesigner(std::size_t sampleFrequency, const std::vector<double>& centerFrequencies);
        virtual ~GraphicEqDesigner();

        DECLARE_NOT_COPYABLE(GraphicEqDesigner);
        DECLARE_NOT_MOVABLE(GraphicEqDesigner);

        void update(const std::vector<double>& gainsDb);
        void update(const std::vector<BiquadCoefficients<T>>& biquadCoefficients, T d0);
        const std::vector<BiquadCoefficients<T>>& biquadCoefficients() const;
        T d0() const;
    };

    template<class T>
    inline
    GraphicEqDesigner<T>::GraphicEqDesigner(std::size_t sampleFrequency, const std::vector<double>& centerFrequencies)
    {
        m_graphicEqDesignerPrivate = move(createGraphicEqDesignerPrivate(sampleFrequency, centerFrequencies));
    }

    template<class T>
    inline GraphicEqDesigner<T>::~GraphicEqDesigner()
    {
    }

    template<class T>
    inline void GraphicEqDesigner<T>::update(const std::vector<double>& gainsDb)
    {
        m_graphicEqDesignerPrivate->update(gainsDb);
    }

    template<class T>
    inline void GraphicEqDesigner<T>::update(const std::vector<BiquadCoefficients<T>>& biquadCoefficients, T d0)
    {
        m_graphicEqDesignerPrivate->update(biquadCoefficients, d0);
    }

    template<class T>
    inline const std::vector<BiquadCoefficients<T>>& GraphicEqDesigner<T>::biquadCoefficients() const
    {
        static_assert(AlwaysFalse<T>::value, "Not supported type");
    }

    template<>
    inline const std::vector<BiquadCoefficients<float>>& GraphicEqDesigner<float>::biquadCoefficients() const
    {
        return m_graphicEqDesignerPrivate->floatBiquadCoefficients();
    }

    template<>
    inline const std::vector<BiquadCoefficients<double>>& GraphicEqDesigner<double>::biquadCoefficients() const
    {
        return m_graphicEqDesignerPrivate->doubleBiquadCoefficients();
    }

    template<class T>
    inline T GraphicEqDesigner<T>::d0() const
    {
        return static_cast<T>(m_graphicEqDesignerPrivate->d0());
    }
}

#endif
