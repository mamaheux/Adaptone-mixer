#ifndef UNIFORMIZATION_AUTOPOSITION_H
#define UNIFORMIZATION_AUTOPOSITION_H

#include "Uniformization/Math.h"

namespace adaptone
{
    class AutoPosition
    {
        double m_alpha;
        double m_espilonTotalDistError;
        double m_espilonDeltaTotalDistError;

        int m_iterNb;
        int m_thermalIterNb;
        int m_tryNb;
        int m_countThreshold;

    public:
        AutoPosition();
        ~AutoPosition();
    };
}
#endif
