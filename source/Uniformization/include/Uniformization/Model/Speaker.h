#ifndef UNIFORMIZATION_MODEL_SPEAKER_H
#define UNIFORMIZATION_MODEL_SPEAKER_H

#include <Uniformization/Model/ModelElement.h>

namespace adaptone
{
    class Speaker : public ModelElement
    {
    public:
        Speaker();
        Speaker(double x, double y, uint32_t id);
        Speaker(double x, double y, double z, uint32_t id);
        ~Speaker() override;
    };
}

#endif
