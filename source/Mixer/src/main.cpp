#include <Mixer/Mixer.h>

#include <iostream>

#if defined(_WIN32) || defined(_WIN64)

#include <windows.h>

static adaptone::Mixer* globalMixer;

BOOL WINAPI consoleHandler(DWORD signal)
{
    if (signal == CTRL_C_EVENT)
    {
        globalMixer->stop();
        std::cout << "Interrupt signal" << std::endl;
    }

    return TRUE;
}

bool setupInterruptSignalHandler(adaptone::Mixer& mixer)
{
    globalMixer = &mixer;
    return static_cast<bool>(SetConsoleCtrlHandler(consoleHandler, TRUE));
}

#elif defined(__unix__) || defined(__linux__)

#include <signal.h>

static adaptone::Mixer* globalMixer;

void my_handler(int s)
{
    globalMixer->stop();
    std::cout << "Interrupt signal" << std::endl;
}

bool setupInterruptSignalHandler(adaptone::Mixer& mixer)
{
    globalMixer = &mixer;

    struct sigaction sigIntHandler;

    sigIntHandler.sa_handler = my_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    return sigaction(SIGINT, &sigIntHandler, NULL) == 0;
}

#endif

using namespace adaptone;

int main(int argc, char** argv)
{
    Configuration configuration(Properties("resources/configuration.properties"));
    Mixer mixer(configuration);

    if (setupInterruptSignalHandler(mixer))
    {
        return mixer.run();
    }

    return -1;
}
