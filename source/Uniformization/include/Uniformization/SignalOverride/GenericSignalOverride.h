#ifndef UNIFORMIZATION_SIGNAL_OVERRIDE_GENERIC_SIGNAL_OVERRIDE_H
#define UNIFORMIZATION_SIGNAL_OVERRIDE_GENERIC_SIGNAL_OVERRIDE_H

#include <Uniformization/SignalOverride/SpecificSignalOverride.h>

#include <Utils/Exception/InvalidValueException.h>

#include <vector>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <typeinfo>
#include <typeindex>

namespace adaptone
{
    class GenericSignalOverride
    {
        std::vector<std::unique_ptr<SpecificSignalOverride>> m_signalSignalOverrides;
        std::unordered_map<std::type_index, size_t> m_indexByType;
        std::atomic<size_t> m_currentSignalOverrideType;

    public:
        GenericSignalOverride(std::vector<std::unique_ptr<SpecificSignalOverride>>&& signalSignalOverrides);
        virtual ~GenericSignalOverride();

        template<class T>
        void setCurrentSignalOverrideType();

        template<class T>
        T& getSignalOverride();

        const PcmAudioFrame& override(const PcmAudioFrame& frame);
    };

    template<class T>
    void GenericSignalOverride::setCurrentSignalOverrideType()
    {
        auto it = m_indexByType.find(typeid(T));
        if (it == m_indexByType.end())
        {
            THROW_INVALID_VALUE_EXCEPTION("currentOverrideType", "");
        }

        m_currentSignalOverrideType.store(it->second);
    }

    template<class T>
    T& GenericSignalOverride::getSignalOverride()
    {
        auto it = m_indexByType.find(typeid(T));
        if (it == m_indexByType.end())
        {
            THROW_INVALID_VALUE_EXCEPTION("currentOverrideType", "");
        }

        return dynamic_cast<T&>(*m_signalSignalOverrides[it->second]);
    }

    inline const PcmAudioFrame& GenericSignalOverride::override(const PcmAudioFrame& frame)
    {
        return m_signalSignalOverrides[m_currentSignalOverrideType.load()]->override(frame);
    }
}

#endif
