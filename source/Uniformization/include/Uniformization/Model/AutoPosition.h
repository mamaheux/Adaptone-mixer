#ifndef UNIFORMIZATION_MODEL_AUTOPOSITION_H
#define UNIFORMIZATION_MODEL_AUTOPOSITION_H

#include <Uniformization/Math.h>
#include <Uniformization/Model/Room.h>

#include <armadillo>

namespace adaptone
{
    class AutoPosition
    {
        double m_alpha;
        double m_epsilonTotalDistanceError;
        double m_epsilonDeltaTotalDistanceError;
        double m_distanceRelativeError;
        std::size_t m_iterationCount;
        std::size_t m_thermalIterationCount;
        std::size_t m_tryCount;
        std::size_t m_countThreshold;

    public:
        AutoPosition(double alpha,
            double epsilonTotalDistanceError,
            double epsilonDeltaTotalDistanceError,
            double distanceRelativeError,
            std::size_t iterationCount,
            std::size_t thermalIterationCount,
            std::size_t tryCount,
            std::size_t countThreshold);
        virtual ~AutoPosition();

        void computeRoomConfiguration2D(Room& room, const arma::mat& distancesMat, bool randomInitConfiguration);
    };
}

#endif
