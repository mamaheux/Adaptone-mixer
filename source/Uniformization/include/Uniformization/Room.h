#ifndef UNIFORMIZATION_ROOM_H
#define UNIFORMIZATION_ROOM_H

#include "Uniformization/Probe.h"
#include "Uniformization/Speaker.h"

#include <vector>
#include <memory>

namespace adaptone
{
    class Room
    {
        std::vector<std::unique_ptr<Probe>> m_probes;
        std::vector<std::unique_ptr<Speaker>> m_speakers;

    public:
        Room();
        ~Room();
    };
}
#endif