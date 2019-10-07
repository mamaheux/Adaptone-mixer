#ifndef SIGNAL_PROCESSING_PARAMETERS_REALTIME_PARAMETERS_H
#define SIGNAL_PROCESSING_PARAMETERS_REALTIME_PARAMETERS_H

#include <Utils/ClassMacro.h>

#include <functional>
#include <mutex>

namespace adaptone
{
    class RealtimeParameters
    {
        bool m_isDirty;
        std::mutex m_mutex;

    public:
        RealtimeParameters(bool isDirty = false);
        virtual ~RealtimeParameters();

        DECLARE_NOT_COPYABLE(RealtimeParameters);
        DECLARE_NOT_MOVABLE(RealtimeParameters);

        bool isDirty();
        void update(const std::function<void()>& function);

        void applyUpdate(const std::function<void()>& function);
        bool tryApplyingUpdate(const std::function<void()>& function);
    };

    inline bool RealtimeParameters::isDirty()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_isDirty;
    }

    inline void RealtimeParameters::update(const std::function<void()>& function)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        function();
        m_isDirty = true;
    }

    inline void RealtimeParameters::applyUpdate(const std::function<void()>& function)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_isDirty)
        {
            function();
            m_isDirty = false;
        }
    }

    inline bool RealtimeParameters::tryApplyingUpdate(const std::function<void()>& function)
    {
        std::unique_lock<std::mutex> lock(m_mutex, std::try_to_lock);
        if (lock.owns_lock())
        {
            if (m_isDirty)
            {
                function();
                m_isDirty = false;
            }
            return true;
        }
        return false;
    }
}

#endif
