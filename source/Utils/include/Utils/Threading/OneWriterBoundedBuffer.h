#ifndef UTILS_THREADING_ONE_WRITER_BOUNDED_BUFFER_H
#define UTILS_THREADING_ONE_WRITER_BOUNDED_BUFFER_H

#include <Utils/Threading/BoundedBuffer.h>

namespace adaptone
{
    template<class T, class S>
    class BasicOneWriterBoundedBuffer : public BasicBoundedBuffer<T, S>
    {
    public:
        BasicOneWriterBoundedBuffer(std::size_t count);
        BasicOneWriterBoundedBuffer(std::size_t count, std::function<T()> initializer);
        BasicOneWriterBoundedBuffer(std::size_t count, std::function<T()> initializer, std::function<void(T&)> deleter);
        ~BasicOneWriterBoundedBuffer() override;

        void writePartialData(std::function<void(T&)> writeFunction);
        void finishWriting();
    };

    template<class T, class S>
    inline BasicOneWriterBoundedBuffer<T, S>::BasicOneWriterBoundedBuffer(std::size_t count) :
        BasicBoundedBuffer<T, S>(count)
    {
    }

    template<class T, class S>
    inline BasicOneWriterBoundedBuffer<T, S>::BasicOneWriterBoundedBuffer(std::size_t count,
        std::function<T()> initializer) : BasicBoundedBuffer<T, S>(count, initializer)
    {
    }

    template<class T, class S>
    inline BasicOneWriterBoundedBuffer<T, S>::BasicOneWriterBoundedBuffer(std::size_t count,
        std::function<T()> initializer, std::function<void(T&)> deleter) :
        BasicBoundedBuffer<T, S>(count, initializer, deleter)
    {
    }

    template<class T, class S>
    inline BasicOneWriterBoundedBuffer<T, S>::~BasicOneWriterBoundedBuffer()
    {
    }

    template<class T, class S>
    void BasicOneWriterBoundedBuffer<T, S>::writePartialData(std::function<void(T&)> writeFunction)
    {
        BasicBoundedBuffer<T, S>::m_empty.wait();
        BasicBoundedBuffer<T, S>::m_mutex.wait();
        T& buffer = BasicBoundedBuffer<T, S>::m_buffers[BasicBoundedBuffer<T, S>::m_writeIndex];
        BasicBoundedBuffer<T, S>::m_mutex.notify();

        try
        {
            writeFunction(buffer);
        }
        catch (...)
        {
            BasicBoundedBuffer<T, S>::m_empty.notify();
            throw;
        }
        BasicBoundedBuffer<T, S>::m_empty.notify();
    }

    template<class T, class S>
    void BasicOneWriterBoundedBuffer<T, S>::finishWriting()
    {
        BasicBoundedBuffer<T, S>::m_empty.wait();
        BasicBoundedBuffer<T, S>::m_mutex.wait();

        BasicBoundedBuffer<T, S>::m_writeIndex =
            (BasicBoundedBuffer<T, S>::m_writeIndex + 1) % BasicBoundedBuffer<T, S>::m_buffers.size();

        BasicBoundedBuffer<T, S>::m_mutex.notify();
        BasicBoundedBuffer<T, S>::m_full.notify();
    }

    template<class T>
    using OneWriterBoundedBuffer = BasicOneWriterBoundedBuffer<T, Semaphore>;

    template<class T>
    using SpinLockOneWriterBoundedBuffer = BasicOneWriterBoundedBuffer<T, SpinLockSemaphore>;
}

#endif
