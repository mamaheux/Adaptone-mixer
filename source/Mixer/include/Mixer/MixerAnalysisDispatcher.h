#ifndef MIXER_MIXER_ANALYSIS_DISPATCHER_H
#define MIXER_MIXER_ANALYSIS_DISPATCHER_H

#include <SignalProcessing/AnalysisDispatcher.h>
#include <SignalProcessing/ProcessingDataType.h>

#include <Communication/Messages/Output/SoundLevelMessage.h>

#include <Utils/ClassMacro.h>
#include <Utils/Logger/Logger.h>
#include <Utils/Threading/BoundedBuffer.h>

#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include <thread>

namespace adaptone
{
    class MixerAnalysisDispatcher : public AnalysisDispatcher
    {
        static constexpr std::size_t SoundLevelBoundedBufferSize = 5;
        static constexpr std::size_t InputSampleBoundedBufferSize = 5;

        std::shared_ptr<Logger> m_logger;
        std::function<void(const ApplicationMessage&)> m_send;
        ProcessingDataType m_processingDataType;

        std::atomic<bool> m_stopped;

        std::unique_ptr<std::thread> m_soundLevelThread;
        BoundedBuffer<SoundLevelMessage> m_soundLevelBoundedBuffer;

        std::unique_ptr<std::thread> m_inputSampleThread;
        BoundedBuffer<float*> m_floatInputSampleBoundedBuffer;
        BoundedBuffer<double*> m_doubleInputSampleBoundedBuffer;

    public:
        MixerAnalysisDispatcher(std::shared_ptr<Logger> logger,
            std::function<void(const ApplicationMessage&)> send,
            ProcessingDataType processingDataType,
            std::size_t frameSampleCount,
            std::size_t sampleFrequency,
            std::size_t inputChannelCount);
        ~MixerAnalysisDispatcher() override;

        DECLARE_NOT_COPYABLE(MixerAnalysisDispatcher);
        DECLARE_NOT_MOVABLE(MixerAnalysisDispatcher);

        void start() override;
        void stop() override;

        void notifySoundLevel(const std::map<SoundLevelType, std::vector<float>>& soundLevels) override;
        void notifySoundLevel(const std::map<SoundLevelType, std::vector<double>>& soundLevels) override;

        void notifyInputEqOutputFrame(const std::function<void(float*)> notifyFunction) override;
        void notifyInputEqOutputFrame(const std::function<void(double*)> notifyFunction) override;

    private:
        void soundLevelRun();
        void floatInputSampleRun();
        void doubleInputSampleRun();

        void startSoundLevelThread();
        void startInputSampleThread();

        void stopSoundLevelThread();
        void stopInputSampleThread();
    };
}

#endif
