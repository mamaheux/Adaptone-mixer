#ifndef UTILS_CONFIGURATION_PROPERTIES_H
#define UTILS_CONFIGURATION_PROPERTIES_H

#include <Utils/ClassMacro.h>
#include <Utils/TypeTraits.h>
#include <Utils/StringUtils.h>
#include <Utils/Exception/PropertyNotFoundException.h>
#include <Utils/Exception/PropertyParseException.h>

#include <unordered_map>
#include <vector>
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
    struct ValueParser
    {
        static T parse(const std::string& key, const std::string& valueStr)
        {
            std::istringstream ss(valueStr);

            T value;
            ss >> value;

            if (ss.fail())
            {
                THROW_PROPERTY_PARSE_EXCEPTION(key, valueStr);
            }

            return value;
        }
    };

    template<>
    struct ValueParser<bool>
    {
        static bool parse(const std::string& key, const std::string& valueStr)
        {
            return valueStr == "true";
        }
    };

    template<>
    struct ValueParser<std::string>
    {
        static std::string parse(const std::string& key, const std::string& valueStr)
        {
            return valueStr;
        }
    };

    template<>
    template<class T>
    struct ValueParser<std::vector<T>>
    {
        static std::vector<T> parse(const std::string& key, const std::string& valueStr)
        {
            if (valueStr.size() < 2 || valueStr[0] != '[' || valueStr[valueStr.size() - 1] != ']')
            {
                THROW_PROPERTY_PARSE_EXCEPTION(key, valueStr);
            }

            std::vector<T> values;
            std::string arrayValue;
            std::stringstream arrayValuesStream(valueStr.substr(1, valueStr.size() - 2));
            while (std::getline(arrayValuesStream, arrayValue, ','))
            {
                trim(arrayValue);
                if (arrayValue != "" || !arrayValuesStream.eof())
                {
                    values.push_back(ValueParser<T>::parse(key, arrayValue));
                }
            }

            return values;
        }
    };

    template<class T>
    inline T Properties::get(const std::string& key) const
    {
        auto it = m_properties.find(key);
        if (it == m_properties.end())
        {
            THROW_PROPERTY_NOT_FOUND_EXCEPTION(key);
        }

        return ValueParser<T>::parse(key, it->second);
    }
}

#endif
