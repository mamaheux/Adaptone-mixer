#ifndef UTILS_DATA_AUDIO_FRAME_H
#define UTILS_DATA_AUDIO_FRAME_H

#include <cstddef>
#include <cstring>

namespace adaptone
{
    /*
     * A templated PCM audio frame
     */
    template <class T>
    class AudioFrame
    {
        std::size_t m_channelCount;
        std::size_t m_sampleCount;
        T* m_data;

    public:
        AudioFrame(std::size_t channelCount, std::size_t sampleCount);
        AudioFrame(const AudioFrame& other);
        AudioFrame(AudioFrame&& other);
        ~AudioFrame();

        std::size_t channelCount() const;
        std::size_t sampleCount() const;

        T* data();
        std::size_t size() const;
        std::size_t byteSize() const;
        T& operator[](std::size_t i);

        AudioFrame& operator=(const AudioFrame& other);
        AudioFrame& operator=(AudioFrame&& other);
    };

    template <class T>
    inline AudioFrame<T>::AudioFrame(std::size_t channelCount, std::size_t sampleCount) :
        m_channelCount(channelCount), m_sampleCount(sampleCount)
    {
        m_data = new T[m_channelCount * m_sampleCount];
    }

    template <class T>
    inline AudioFrame<T>::AudioFrame(const AudioFrame<T>& other) :
        m_channelCount(other.m_channelCount), m_sampleCount(other.m_sampleCount)
    {
        m_data = new T[m_channelCount * m_sampleCount];
        std::memcpy(m_data, other.m_data, m_channelCount * m_sampleCount * sizeof(T));
    }

    template <class T>
    inline AudioFrame<T>::AudioFrame(AudioFrame<T>&& other):
        m_channelCount(other.m_channelCount), m_sampleCount(other.m_sampleCount)
    {
        m_data = other.m_data;

        other.m_channelCount = 0;
        other.m_sampleCount = 0;
        other.m_data = nullptr;
    }

    template <class T>
    inline AudioFrame<T>::~AudioFrame()
    {
        if (m_data != nullptr)
        {
            delete[] m_data;
        }
    }

    template <class T>
    inline std::size_t AudioFrame<T>::channelCount() const
    {
        return m_channelCount;
    }

    template <class T>
    inline std::size_t AudioFrame<T>::sampleCount() const
    {
        return m_sampleCount;
    }

    template <class T>
    inline T* AudioFrame<T>::data()
    {
        return m_data;
    }

    template <class T>
    inline std::size_t AudioFrame<T>::size() const
    {
        return m_channelCount * m_sampleCount;
    }

    template <class T>
    inline std::size_t AudioFrame<T>::byteSize() const
    {
        return m_channelCount * m_sampleCount * sizeof(T);
    }

    template <class T>
    inline T& AudioFrame<T>::operator[](std::size_t i)
    {
        return m_data[i];
    }

    template <class T>
    inline AudioFrame<T>& AudioFrame<T>::operator=(const AudioFrame<T>& other)
    {
        if (m_data != nullptr)
        {
            delete[] m_data;
        }

        m_channelCount = other.m_channelCount;
        m_sampleCount = other.m_sampleCount;

        m_data = new T[m_channelCount * m_sampleCount];
        std::memcpy(m_data, other.m_data, m_channelCount * m_sampleCount * sizeof(T));
    }

    template <class T>
    inline AudioFrame<T>& AudioFrame<T>::operator=(AudioFrame<T>&& other)
    {
        if (m_data != nullptr)
        {
            delete[] m_data;
        }

        m_channelCount = other.m_channelCount;
        m_sampleCount = other.m_sampleCount;
        m_data = other.m_data;

        other.m_channelCount = 0;
        other.m_sampleCount = 0;
        other.m_data = nullptr;
    }
}

#endif