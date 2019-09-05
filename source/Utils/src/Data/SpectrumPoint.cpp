#include <Utils/Data/SpectrumPoint.h>

using namespace adaptone;
using namespace std;

SpectrumPoint::SpectrumPoint() : m_frequency(0), m_amplitude(0)
{
}

SpectrumPoint::SpectrumPoint(double frequency, double amplitude) : m_frequency(frequency), m_amplitude(amplitude)
{
}

SpectrumPoint::~SpectrumPoint()
{
}
