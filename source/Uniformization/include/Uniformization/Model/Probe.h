#ifndef UNIFORMIZATION_MODEL_PROBE_H
#define UNIFORMIZATION_MODEL_PROBE_H

#include <Uniformization/Model/ModelElement.h>

namespace adaptone
{
    class Probe : public ModelElement
    {
    public:
        Probe();
        Probe(double x, double y, uint32_t id);
        Probe(double x, double y, double z, uint32_t id);
        ~Probe() override;
    };
}

#endif
