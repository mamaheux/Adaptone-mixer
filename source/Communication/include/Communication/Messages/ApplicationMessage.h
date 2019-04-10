#ifndef COMMUNICATION_MESAGES_APPLICATION_MESSAGE_H
#define COMMUNICATION_MESAGES_APPLICATION_MESSAGE_H

#include <nlohmann/json.hpp>

namespace adaptone
{
    class ApplicationMessage
    {
    public:
        ApplicationMessage();
        virtual ~ApplicationMessage();

        virtual std::size_t seqId() const = 0;
    };
}

#endif
