#ifndef COMMUNICATION_MESAGES_INITIALIZATION_SYMMETRY_CONFIRMATION_MESSAGE_H
#define COMMUNICATION_MESAGES_INITIALIZATION_SYMMETRY_CONFIRMATION_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>

#include <cstddef>

namespace adaptone
{
    class SymmetryConfirmationMessage : public ApplicationMessage
    {
        std::size_t m_symmetry;

    public:
        SymmetryConfirmationMessage();
        SymmetryConfirmationMessage(std::size_t symmetry);
        virtual ~SymmetryConfirmationMessage();

        std::size_t symmetry() const;

        friend void to_json(nlohmann::json& j, const SymmetryConfirmationMessage& o);
        friend void from_json(const nlohmann::json& j, SymmetryConfirmationMessage& o);
    };

    inline std::size_t SymmetryConfirmationMessage::symmetry() const
    {
        return m_symmetry;
    }

    inline void to_json(nlohmann::json& j, const SymmetryConfirmationMessage& o)
    {
        nlohmann::json data({{ "symmetry", o.m_symmetry }});
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, SymmetryConfirmationMessage& o)
    {
        j.at("data").at("symmetry").get_to(o.m_symmetry);
    }
}

#endif
