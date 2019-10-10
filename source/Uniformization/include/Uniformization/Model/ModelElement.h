#ifndef UNIFORMIZATION_MODEL_MODEL_ELEMENT_H
#define UNIFORMIZATION_MODEL_MODEL_ELEMENT_H

namespace adaptone
{
    class ModelElement
    {
        double m_x;
        double m_y;
        double m_z;

    public:
        ModelElement();
        ModelElement(double x, double y);
        ModelElement(double x, double y, double z);
        virtual ~ModelElement();

        void setX(double x);
        void setY(double y);
        void setZ(double z);

        double x() const;
        double y() const;
        double z() const;
    };

    inline void ModelElement::setX(double x)
    {
        m_x = x;
    }

    inline void ModelElement::setY(double y)
    {
        m_y = y;
    }

    inline void ModelElement::setZ(double z)
    {
        m_z = z;
    }

    inline double ModelElement::x() const
    {
        return m_x;
    }

    inline double ModelElement::y() const
    {
        return m_y;
    }

    inline double ModelElement::z() const
    {
        return m_z;
    }
}

#endif
