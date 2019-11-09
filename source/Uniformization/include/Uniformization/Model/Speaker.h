#ifndef UNIFORMIZATION_MODEL_SPEAKER_H
#define UNIFORMIZATION_MODEL_SPEAKER_H

#include <Uniformization/Model/ModelElement.h>

#include <armadillo>

namespace adaptone
{
    class Speaker : public ModelElement
    {
        arma::mat m_directivities;

    public:
        Speaker();
        Speaker(double x, double y, uint32_t id);
        Speaker(double x, double y, double z, uint32_t id);
        ~Speaker() override;

        void setDirectivities(const arma::mat& directivities);
        const arma::mat& directivities() const;
    };

    inline void Speaker::setDirectivities(const arma::mat& directivities)
    {
        m_directivities = directivities;
    }

    inline const arma::mat& Speaker::directivities() const
    {
        return m_directivities;
    }
}

#endif
