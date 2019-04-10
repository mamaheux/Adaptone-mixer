#ifndef COMMUNICATION_MESAGES_APPLICATION_MESSAGE_H
#define COMMUNICATION_MESAGES_APPLICATION_MESSAGE_H

#include <nlohmann/json.hpp>

namespace adaptone
{
    class ApplicationMessage
    {
        std::size_t m_seqId;

    public:
        ApplicationMessage(std::size_t seqId);
        virtual ~ApplicationMessage();

        std::size_t seqId() const;
    };

    inline std::size_t ApplicationMessage::seqId() const
    {
        return m_seqId;
    }
}

#endif
