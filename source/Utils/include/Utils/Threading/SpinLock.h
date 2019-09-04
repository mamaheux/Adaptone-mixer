#ifndef UTILS_THREADING_SPIN_LOCK_H
#define UTILS_THREADING_SPIN_LOCK_H

#include <Utils/ClassMacro.h>

#include <atomic>

namespace adaptone
{
    class SpinLock
    {
        std::atomic_flag m_flag;

    public:
        SpinLock();
        virtual ~SpinLock();

        DECLARE_NOT_COPYABLE(SpinLock);
        DECLARE_NOT_MOVABLE(SpinLock);

        void lock();
        void unlock();
    };

    inline void SpinLock::lock()
    {
        while (m_flag.test_and_set(std::memory_order_acquire))
        {
        }
    }

    inline void SpinLock::unlock()
    {
        m_flag.clear(std::memory_order_release);
    }
}

#endif
