#include <Uniformization/Communication/Messages/ProbeMessage.h>

#include <unordered_map>

#include <iostream>

using namespace adaptone;
using namespace std;

ProbeMessage::ProbeMessage(uint32_t id, size_t payloadSize) : m_id(id), m_fullSize(payloadSize + sizeof(m_id))
{
}

ProbeMessage::~ProbeMessage()
{
}

bool ProbeMessage::hasPayload(uint32_t id)
{
    static const unordered_map<uint32_t, bool> Mapping(
        {
            { 0, false },
            { 1, false },
            { 2, true },
            { 3, true },
            { 4, false },
            { 5, true },
            { 6, true },
            { 7, true }
        });

    auto it = Mapping.find(id);
    if (it != Mapping.end())
    {
        return it->second;
    }

    cout << "hasPayload::id" << id << endl;
    THROW_INVALID_VALUE_EXCEPTION("Format not supported", "");
}
