#ifndef UNIFORMIZATION_MODEL_MODEL_ELEMENT_H
#define UNIFORMIZATION_MODEL_MODEL_ELEMENT_H

#include <cstdint>

namespace adaptone
{
    class ModelElement
    {
        double m_x;
        double m_y;
        double m_z;

        uint32_t m_id;

    public:
        ModelElement();
        ModelElement(double x, double y, uint32_t id);
        ModelElement(double x, double y, double z, uint32_t id);
        virtual ~ModelElement();

        void setX(double x);
        void setY(double y);
        void setZ(double z);

        double x() const;
        double y() const;
        double z() const;

        void setId(uint32_t id);
        uint32_t id() const;
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

    inline void ModelElement::setId(uint32_t id)
    {
        m_id = id;
    }

    inline uint32_t ModelElement::id() const
    {
        return m_id;
    }
}

#endif
