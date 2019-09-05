#ifndef UTILS_THREADING_BOUNDED_BUFFER_H
#define UTILS_THREADING_BOUNDED_BUFFER_H

#include <Utils/ClassMacro.h>
#include <Utils/Threading/Semaphore.h>
#include <Utils/Threading/SpinLockSemaphore.h>

#include <vector>
#include <functional>

namespace adaptone
{
    template<class T, class S>
    class BasicBoundedBuffer
    {
    protected:
        std::vector<T> m_buffers;
        S m_mutex;
        S m_empty;
        S m_full;

        std::size_t m_readIndex;
        std::size_t m_writeIndex;

        std::function<void(T&)> m_deleter;

    public:
        BasicBoundedBuffer(std::size_t count);
        BasicBoundedBuffer(std::size_t count, std::function<T()> initializer);
        BasicBoundedBuffer(std::size_t count, std::function<T()> initializer, std::function<void(T&)> deleter);
        virtual ~BasicBoundedBuffer();

        void read(std::function<void(const T&)> callback);
        void write(std::function<void(T&)> callback);

        std::vector<T> buffers();
    };

    template<class T, class S>
    inline BasicBoundedBuffer<T, S>::BasicBoundedBuffer(std::size_t count) :
        m_buffers(count), m_mutex(1), m_empty(count), m_full(0), m_readIndex(0), m_writeIndex(0)
    {
        m_deleter = [](T& o) {};
    }

    template<class T, class S>
    inline BasicBoundedBuffer<T, S>::BasicBoundedBuffer(std::size_t count, std::function<T()> initializer) :
        BasicBoundedBuffer(count)
    {
        for (std::size_t i = 0; i < m_buffers.size(); i++)
        {
            m_buffers[i] = initializer();
        }
    }

    template<class T, class S>
    inline BasicBoundedBuffer<T, S>::BasicBoundedBuffer(std::size_t count, std::function<T()> initializer,
        std::function<void(T&)> deleter) : BasicBoundedBuffer(count, initializer)
    {
        m_deleter = deleter;
    }

    template<class T, class S>
    inline BasicBoundedBuffer<T, S>::~BasicBoundedBuffer()
    {
        for (std::size_t i = 0; i < m_buffers.size(); i++)
        {
            m_deleter(m_buffers[i]);
        }
    }

    template<class T, class S>
    inline void BasicBoundedBuffer<T, S>::read(std::function<void(const T&)> readFunction)
    {
        m_full.wait();
        m_mutex.wait();

        T& buffer = m_buffers[m_readIndex];
        m_readIndex = (m_readIndex + 1) % m_buffers.size();
        m_mutex.notify();

        try
        {
            readFunction(buffer);
        }
        catch (...)
        {
            m_empty.notify();
            throw;
        }
        m_empty.notify();
    }

    template<class T, class S>
    inline void BasicBoundedBuffer<T, S>::write(std::function<void(T&)> writeFunction)
    {
        m_empty.wait();
        m_mutex.wait();

        T& buffer = m_buffers[m_writeIndex];
        m_writeIndex = (m_writeIndex + 1) % m_buffers.size();
        m_mutex.notify();

        try
        {
            writeFunction(buffer);
        }
        catch (...)
        {
            m_full.notify();
            throw;
        }
        m_full.notify();
    }

    template<class T, class S>
    inline std::vector<T> BasicBoundedBuffer<T, S>::buffers()
    {
        return m_buffers;
    };

    template<class T>
    using BoundedBuffer = BasicBoundedBuffer<T, Semaphore>;

    template<class T>
    using SpinLockBoundedBuffer = BasicBoundedBuffer<T, SpinLockSemaphore>;
}

#endif
