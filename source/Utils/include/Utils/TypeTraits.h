#ifndef UTILS_TYPE_TRAITS_H
#define UTILS_TYPE_TRAITS_H

#include <type_traits>

namespace adaptone
{
    template<class T>
    struct AlwaysFalse : std::false_type
    {
    };
}

#endif
