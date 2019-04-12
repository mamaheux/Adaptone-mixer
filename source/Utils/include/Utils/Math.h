#ifndef UTILS_MATH_H
#define UTILS_MATH_H

#include <cmath>
#include <vector>
#include <algorithm>

namespace adaptone
{
    template<class T>
    inline T scalarToDb(T value)
    {
        return static_cast<T>(20 * std::log10(value));
    }

    template<class T>
    inline std::vector<T> vectorToDb(const std::vector<T>& values)
    {
        std::vector<double> valuesDb(values.size());
        std::transform(values.begin(), values.end(), valuesDb.begin(), scalarToDb<T>);
        return valuesDb;
    }
}

#endif
