#ifndef UTILS_FUNCTIONAL_FUNCTION_QUEUE_H
#define UTILS_FUNCTIONAL_FUNCTION_QUEUE_H

#include <Utils/ClassMacro.h>
#include <Utils/TypeTraits.h>

#include <functional>
#include <mutex>
#include <queue>
#include <type_traits>

namespace adaptone
{
    template<class T, bool IsReturningBool>
    class FunctionQueueCommon
    {
    protected:
        std::queue<std::function<T>> m_functionQueue;

        FunctionQueueCommon();
        virtual ~FunctionQueueCommon();
        void executeAndRemove();
    };

    template<class T, bool IsReturningBool>
    FunctionQueueCommon<T, IsReturningBool>::FunctionQueueCommon()
    {
    }

    template<class T, bool IsReturningBool>
    FunctionQueueCommon<T, IsReturningBool>::~FunctionQueueCommon()
    {
    }

    template<class T, bool IsReturningBool>
    void FunctionQueueCommon<T, IsReturningBool>::executeAndRemove()
    {
        if (m_functionQueue.size() > 0)
        {
            m_functionQueue.front()();
            m_functionQueue.pop();
        }
    }

    template<class T>
    class FunctionQueueCommon<T, true>
    {
    protected:
        std::queue<std::function<T>> m_functionQueue;
        std::mutex m_mutex;

        FunctionQueueCommon();
        virtual ~FunctionQueueCommon();
        void executeAndRemove();
    };

    template<class T>
    FunctionQueueCommon<T, true>::FunctionQueueCommon()
    {
    }

    template<class T>
    FunctionQueueCommon<T, true>::~FunctionQueueCommon()
    {
    }

    template<class T>
    void FunctionQueueCommon<T, true>::executeAndRemove()
    {
        if (m_functionQueue.size() > 0 && m_functionQueue.front()())
        {
            m_functionQueue.pop();
        }
    }

    template<class T>
    class FunctionQueue : public FunctionQueueCommon<T, IsReturningBool<T>::value>
    {
        typedef FunctionQueueCommon<T, IsReturningBool<T>::value> BaseType;

        std::mutex m_mutex;

    public:
        FunctionQueue();
        virtual ~FunctionQueue();

        DECLARE_NOT_COPYABLE(FunctionQueue);
        DECLARE_NOT_MOVABLE(FunctionQueue);

        void push(const std::function<T>& function);
        void tryExecute();
        void execute();
    };

    template<class T>
    FunctionQueue<T>::FunctionQueue()
    {
    }

    template<class T>
    FunctionQueue<T>::~FunctionQueue()
    {
    }

    template<class T>
    void FunctionQueue<T>::push(const std::function<T>& function)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        BaseType::m_functionQueue.push(function);
    }

    template<class T>
    void FunctionQueue<T>::tryExecute()
    {
        std::unique_lock<std::mutex> lock(m_mutex, std::try_to_lock);
        if (lock.owns_lock())
        {
            BaseType::executeAndRemove();
        }
    }

    template<class T>
    void FunctionQueue<T>::execute()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        BaseType::executeAndRemove();
    }
}

#endif
