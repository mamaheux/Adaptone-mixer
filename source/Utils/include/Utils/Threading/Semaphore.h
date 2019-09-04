#ifndef UTILS_THREADING_SEMAPHORE_H
#define UTILS_THREADING_SEMAPHORE_H

#include <Utils/ClassMacro.h>

#include <condition_variable>
#include <mutex>

namespace adaptone
{
    class Semaphore
    {
        std::mutex m_mutex;
        std::condition_variable m_conditionVariable;
        volatile  std::size_t m_count;

    public:
        Semaphore(std::size_t count);
        virtual ~Semaphore();

        DECLARE_NOT_COPYABLE(Semaphore);
        DECLARE_NOT_MOVABLE(Semaphore);

        void notify();
        void wait();
    };

    inline void Semaphore::notify()
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        m_count++;
        m_conditionVariable.notify_one();
    }

    inline void Semaphore::wait()
    {
        std::unique_lock<decltype(m_mutex)> lock(m_mutex);
        m_conditionVariable.wait(lock, [&] () { return m_count; });
        m_count--;
    }
}

#endif
