#ifndef SIGNAL_PROCESSING_TESTS_CUDA_FREE_GUARD_H
#define SIGNAL_PROCESSING_TESTS_CUDA_FREE_GUARD_H

#include <Utils/ClassMacro.h>

namespace adaptone
{
    class CudaFreeGuard
    {
        void* m_pointer;
    public:
        CudaFreeGuard(void* pointer);
        virtual ~CudaFreeGuard();

        DECLARE_NOT_COPYABLE(CudaFreeGuard);
        DECLARE_NOT_MOVABLE(CudaFreeGuard);
    };

    CudaFreeGuard::CudaFreeGuard(void* pointer) : m_pointer(pointer)
    {
    }

    CudaFreeGuard::~CudaFreeGuard()
    {
        cudaFree(m_pointer);
    }
}

#endif
