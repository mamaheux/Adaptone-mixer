#include <Mixer/Mixer.h>

using namespace adaptone;

int main(int argc, char** argv)
{
    Configuration configuration(Properties("resources/configuration.properties"));
    Mixer mixer(configuration);

    return mixer.run();
}