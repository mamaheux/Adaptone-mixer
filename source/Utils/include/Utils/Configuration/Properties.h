#ifndef UTILS_CONFIGURATION_PROPERTIES_H
#define UTILS_CONFIGURATION_PROPERTIES_H

#include <Utils/ClassMacro.h>
#include <Utils/TypeTraits.h>
#include <Utils/Exception/PropertyNotFoundException.h>
#include <Utils/Exception/PropertyParseException.h>

#include <unordered_map>
#include <string>
#include <istream>
#include <sstream>

namespace adaptone
{
    class Properties
    {
        std::unordered_map<std::string, std::string> m_properties;

    public:
        Properties(const std::unordered_map<std::string, std::string>& properties);
        explicit Properties(const std::string& filename);
        virtual ~Properties();

        DECLARE_NOT_COPYABLE(Properties);
        DECLARE_NOT_MOVABLE(Properties);

        template<class T>
        T get(const std::string& key) const;

    private:
        void parse(std::istream& stream);
        void parseLine(const std::string& line);
    };

    template<class T>
    inline T Properties::get(const std::string& key) const
    {
        std::string valueStr = get<std::string>(key);

        std::istringstream ss(valueStr);

        T value;
        ss >> value;

        if (ss.fail())
        {
            THROW_PROPERTY_PARSE_EXCEPTION(key, valueStr);
        }

        return value;
    }

    template<>
    inline std::string Properties::get(const std::string& key) const
    {
        auto it = m_properties.find(key);
        if (it == m_properties.end())
        {
            THROW_PROPERTY_NOT_FOUND_EXCEPTION(key);
        }

        return it->second;
    }

    template<>
    inline bool Properties::get(const std::string& key) const
    {
        return get<std::string>(key) == "true";

    }
}

#endif
