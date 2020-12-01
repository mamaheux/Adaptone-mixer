#ifndef COMMUNICATION_MESAGES_INITIALIZATION_POSITION_CONFIRMATION_MESSAGE_H
#define COMMUNICATION_MESAGES_INITIALIZATION_POSITION_CONFIRMATION_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>
#include <Communication/Messages/Initialization/ConfigurationPosition.h>

#include <vector>

namespace adaptone
{
    class PositionConfirmationMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 3;

    private:
        std::vector<ConfigurationPosition> m_firstSymmetryPositions;
        std::vector<ConfigurationPosition> m_secondSymmetryPositions;

    public:
        PositionConfirmationMessage();
        PositionConfirmationMessage(std::vector<ConfigurationPosition> firstSymmetryPositions,
            std::vector<ConfigurationPosition> secondSymmetryPositions);
        ~PositionConfirmationMessage() override;

        const std::vector<ConfigurationPosition>& firstSymmetryPositions() const;
        const std::vector<ConfigurationPosition>& secondSymmetryPositions() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const PositionConfirmationMessage& o);
        friend void from_json(const nlohmann::json& j, PositionConfirmationMessage& o);
    };

    inline const std::vector<ConfigurationPosition>& PositionConfirmationMessage::firstSymmetryPositions() const
    {
        return m_firstSymmetryPositions;
    }

    inline const std::vector<ConfigurationPosition>& PositionConfirmationMessage::secondSymmetryPositions() const
    {
        return m_secondSymmetryPositions;
    }

    inline std::string PositionConfirmationMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const PositionConfirmationMessage& o)
    {
        nlohmann::json data({{ "firstSymmetryPositions", o.m_firstSymmetryPositions },
            { "secondSymmetryPositions", o.m_secondSymmetryPositions },
        });
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, PositionConfirmationMessage& o)
    {
        j.at("data").at("firstSymmetryPositions").get_to(o.m_firstSymmetryPositions);
        j.at("data").at("secondSymmetryPositions").get_to(o.m_secondSymmetryPositions);
    }
}

#endif
