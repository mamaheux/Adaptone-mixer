#ifndef UNIFORMIZATION_MODEL_AUTOPOSITION_H
#define UNIFORMIZATION_MODEL_AUTOPOSITION_H

#include "Uniformization/Math.h"
#include "Room.h"

#include <armadillo>

namespace adaptone
{
    class AutoPosition
    {
        double m_alpha;
        double m_epsilonTotalDistanceError;
        double m_epsilonDeltaTotalDistanceError;

        std::size_t m_iterCount;
        std::size_t m_thermalIterCount;
        std::size_t m_tryCount;
        std::size_t m_countThreshold;

    public:
        AutoPosition();
        AutoPosition(double alpha, double epsilonTotalDistanceError, double epsilonDeltaTotalDistanceError,
            std::size_t iterCount, std::size_t thermalIterCount, std::size_t tryCount, std::size_t countThreshold);
        ~AutoPosition();

        void computeRoomConfiguration2D(Room& room, const arma::mat& distancesMat, float distanceRelativeError,
            bool randomInitConfiguration);
    };
}

#endif
