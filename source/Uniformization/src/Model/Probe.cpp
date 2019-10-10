#include <Uniformization/Model/Probe.h>

using namespace adaptone;

Probe::Probe()
{
}

Probe::Probe(double x, double y) : ModelElement(x, y)
{
}

Probe::Probe(double x, double y, double z) : ModelElement(x, y, z)
{
}

Probe::~Probe()
{
}
