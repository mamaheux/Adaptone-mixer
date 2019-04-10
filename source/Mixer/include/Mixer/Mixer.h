#ifndef MIXER_MIXER_H
#define MIXER_MIXER_H

#include <Mixer/Configuration/Configuration.h>
#include <Mixer/AudioInput/AudioInput.h>
#include <Mixer/AudioOutput/AudioOutput.h>

#include <Utils/ClassMacro.h>
#include <Utils/Logger/Logger.h>

#include <SignalProcessing/SignalProcessor.h>
#include <SignalProcessing/AnalysisDispatcher.h>

#include <memory>
#include <atomic>
#include <thread>

namespace adaptone
{
    class Mixer
    {
        Configuration m_configuration;
        std::shared_ptr<Logger> m_logger;

        std::unique_ptr<AudioInput> m_audioInput;
        std::unique_ptr<AudioOutput> m_audioOutput;

        std::unique_ptr<SignalProcessor> m_signalProcessor;

        std::shared_ptr<AnalysisDispatcher> m_analysisDispatcher;

        std::unique_ptr<std::thread> m_analysisThread;
        std::atomic<bool> m_stopped;

    public:
        Mixer(const Configuration& configuration);
        virtual ~Mixer();

        DECLARE_NOT_COPYABLE(Mixer);
        DECLARE_NOT_MOVABLE(Mixer);

        void run();
        void stop();

    private:
        std::shared_ptr<Logger> createLogger();

        std::unique_ptr<AudioInput> createAudioInput();
        std::unique_ptr<AudioOutput> createAudioOutput();

        std::unique_ptr<SignalProcessor> createSignalProcessor();

        std::shared_ptr<AnalysisDispatcher> createAnalysisDispatcher();

        void analysisRun();
        void processingRun();
    };
}

#endif
