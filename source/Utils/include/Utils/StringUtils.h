#ifndef UTILS_STRING_UTILS_H
#define UTILS_STRING_UTILS_H

#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>

namespace adaptone
{
    inline std::string& trimLeft(std::string& str)
    {
        str.erase(str.begin(), std::find_if(str.begin(), str.end(),
                                        std::not1(std::ptr_fun<int, int>(std::isspace))));
        return str;
    }

    inline std::string& trimRight(std::string& str)
    {
        str.erase(std::find_if(str.rbegin(), str.rend(),
                             std::not1(std::ptr_fun<int, int>(std::isspace))).base(), str.end());
        return str;
    }

    inline std::string& trim(std::string& str)
    {
        return trimLeft(trimRight(str));
    }
}

#endif