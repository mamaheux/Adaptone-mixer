#ifndef SIGNAL_PROCESSING_PARAMETERS_REALTIME_PARAMETERS_H
#define SIGNAL_PROCESSING_PARAMETERS_REALTIME_PARAMETERS_H

#include <functional>
#include <mutex>

namespace adaptone
{
    class RealtimeParameters
    {
        bool m_isDirty;
        std::mutex m_mutex;

    public:
        RealtimeParameters();
        virtual ~RealtimeParameters();

        bool isDirty();
        void uptate(const std::function<void()>& function);

        void applyUptate(const std::function<void()>& function);
        bool tryApplyingUptate(const std::function<void()>& function);
    };

    inline bool RealtimeParameters::isDirty()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_isDirty;
    }

    inline void RealtimeParameters::uptate(const std::function<void()>& function)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        function();
        m_isDirty = true;
    }

    inline void RealtimeParameters::applyUptate(const std::function<void()>& function)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_isDirty)
        {
            function();
            m_isDirty = false;
        }
    }

    inline bool RealtimeParameters::tryApplyingUptate(const std::function<void()>& function)
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
