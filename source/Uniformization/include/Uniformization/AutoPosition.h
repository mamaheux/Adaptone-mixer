#ifndef UNIFORMIZATION_AUTOPOSITION_H
#define UNIFORMIZATION_AUTOPOSITION_H

#include "Uniformization/Math.h"
#include "Uniformization/Room.h"

#include <armadillo>

namespace adaptone
{
    class AutoPosition
    {
        double m_alpha;
        double m_epsilonTotalDistError;
        double m_epsilonDeltaTotalDistError;

        int m_iterNb;
        int m_thermalIterNb;
        int m_tryNb;
        int m_countThreshold;

    public:
        AutoPosition();
        AutoPosition(double alpha, double epsilonTotalDistError, double epsilonDeltaTotalDistError, int iterNb,
            int thermalIterNb, int tryNb, int countThreshold);
        ~AutoPosition();

        void computeRoomConfiguration2D(Room& room, const arma::mat& distancesMat, float distRelativeError,
            bool randomInitConfig);
    };
}
#endif
