#ifndef UNIFORMIZATION_MODEL_SPEAKER_H
#define UNIFORMIZATION_MODEL_SPEAKER_H

#include <Uniformization/Model/ModelElement.h>

namespace adaptone
{
    class Speaker : public ModelElement
    {
        float m_x;
        float m_y;
        float m_z;

    public:
        Speaker();
        Speaker(double x, double y);
        Speaker(double x, double y, double z);
        ~Speaker() override;
    };
}

#endif
