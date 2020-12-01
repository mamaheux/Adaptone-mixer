#ifndef UNIFORMIZATION_MODEL_SPEAKER_H
#define UNIFORMIZATION_MODEL_SPEAKER_H

#include <Uniformization/Model/ModelElement.h>

#include <armadillo>

namespace adaptone
{
    class Speaker : public ModelElement
    {
        arma::vec m_directivities;

    public:
        Speaker();
        Speaker(double x, double y, uint32_t id);
        Speaker(double x, double y, double z, uint32_t id);
        ~Speaker() override;

        void setDirectivities(arma::vec directivities);
        const arma::vec& directivities() const;
    };

    inline void Speaker::setDirectivities(arma::vec directivities)
    {
        m_directivities = std::move(directivities);
    }

    inline const arma::vec& Speaker::directivities() const
    {
        return m_directivities;
    }
}

#endif
