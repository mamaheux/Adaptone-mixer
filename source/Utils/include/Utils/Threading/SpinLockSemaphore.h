#ifndef UTILS_THREADING_SPIN_LOCK_SEMAPHORE_H
#define UTILS_THREADING_SPIN_LOCK_SEMAPHORE_H

#include <Utils/ClassMacro.h>
#include <Utils/Threading/SpinLock.h>

#include <atomic>
#include <mutex>

namespace adaptone
{
    class SpinLockSemaphore
    {
        volatile int m_count;
        SpinLock m_countSpinLock;
        SpinLock m_waitSpinLock;

    public:
        SpinLockSemaphore(int count);
        virtual ~SpinLockSemaphore();

        DECLARE_NOT_COPYABLE(SpinLockSemaphore);
        DECLARE_NOT_MOVABLE(SpinLockSemaphore);

        void notify();
        void wait();
    };

    inline void SpinLockSemaphore::notify()
    {
        m_countSpinLock.lock();
        m_count++;

        if (m_count <= 0)
        {
            m_waitSpinLock.unlock();
        }
        else
        {
            m_countSpinLock.unlock();
        }
    }

    inline void SpinLockSemaphore::wait()
    {
        m_countSpinLock.lock();

        m_count--;
        if (m_count < 0)
        {
            m_countSpinLock.unlock();
            m_waitSpinLock.lock();
        }
        m_countSpinLock.unlock();
    }
}

#endif
