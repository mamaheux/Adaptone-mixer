#ifndef COMMUNICATION_MESAGES_INITIALIZATION_OPTIMIZED_POSITION_MESSAGE_H
#define COMMUNICATION_MESAGES_INITIALIZATION_OPTIMIZED_POSITION_MESSAGE_H

#include <Communication/Messages/ApplicationMessage.h>
#include <Communication/Messages/Initialization/ConfigurationPosition.h>

#include <vector>

namespace adaptone
{
    class OptimizedPositionMessage : public ApplicationMessage
    {
    public:
        static constexpr std::size_t SeqId = 7;

    private:
        std::vector<ConfigurationPosition> m_positions;

    public:
        OptimizedPositionMessage();
        explicit OptimizedPositionMessage(const std::vector<ConfigurationPosition>& positions);
        ~OptimizedPositionMessage() override;

        const std::vector<ConfigurationPosition>& positions() const;

        std::string toJson() const override;
        friend void to_json(nlohmann::json& j, const OptimizedPositionMessage& o);
        friend void from_json(const nlohmann::json& j, OptimizedPositionMessage& o);
    };

    inline const std::vector<ConfigurationPosition>& OptimizedPositionMessage::positions() const
    {
        return m_positions;
    }

    inline std::string OptimizedPositionMessage::toJson() const
    {
        nlohmann::json serializedMessage = *this;
        return serializedMessage.dump();
    }

    inline void to_json(nlohmann::json& j, const OptimizedPositionMessage& o)
    {
        nlohmann::json data({{ "positions", o.m_positions }});
        j = nlohmann::json{{ "seqId", o.seqId() }, { "data", data }};
    }

    inline void from_json(const nlohmann::json& j, OptimizedPositionMessage& o)
    {
        j.at("data").at("positions").get_to(o.m_positions);
    }
}

#endif
