#include <Utils/Configuration/Properties.h>
#include <Utils/StringUtils.h>

#include <fstream>

using namespace adaptone;
using namespace std;

Properties::Properties(const std::unordered_map<std::string, std::string>& properties) : m_properties(properties)
{
}

Properties::Properties(const std::string& filename)
{
    ifstream fileStream(filename, ifstream::in);
    parse(fileStream);
}

Properties::~Properties()
{
}

void Properties::parse(istream& stream)
{
    string line;
    while (stream.good())
    {
        getline(stream, line);
        parseLine(line);
    }
}

void Properties::parseLine(const std::string& line)
{
    size_t equalIndex = line.find('=');
    size_t hashTagIndex = line.find('#');

    if (equalIndex == string::npos || hashTagIndex < equalIndex)
    { return; }

    string key = line.substr(0, equalIndex);
    string value = line.substr(equalIndex + 1, hashTagIndex - equalIndex - 1);

    trim(key);
    trim(value);

    m_properties[key] = value;
}
