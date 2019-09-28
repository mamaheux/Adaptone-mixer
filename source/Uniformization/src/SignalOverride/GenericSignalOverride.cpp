#include <Uniformization/SignalOverride/GenericSignalOverride.h>

#include <Utils/Exception/InvalidValueException.h>

using namespace adaptone;
using namespace std;

GenericSignalOverride::GenericSignalOverride(const vector<shared_ptr<SpecificSignalOverride>>& signalSignalOverrides) :
    m_signalSignalOverrides(move(signalSignalOverrides)),
    m_currentSignalOverrideType(0)
{
    if (m_signalSignalOverrides.size() == 0)
    {
        THROW_INVALID_VALUE_EXCEPTION("signalSignalOverrides.size()", "");
    }

    for (size_t i = 0; i < m_signalSignalOverrides.size(); i++)
    {
        if (m_indexByType.find(typeid(*m_signalSignalOverrides[i])) != m_indexByType.end())
        {
            THROW_INVALID_VALUE_EXCEPTION("Same types are not supported.", "");
        }

        m_indexByType[typeid(*m_signalSignalOverrides[i])] = i;
    }
}

GenericSignalOverride::~GenericSignalOverride()
{
}
