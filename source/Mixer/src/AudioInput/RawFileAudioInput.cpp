#include <Mixer/AudioInput/RawFileAudioInput.h>

#include <Utils/Exception/NotSupportedException.h>

using namespace adaptone;
using namespace std;

RawFileAudioInput::RawFileAudioInput(PcmAudioFrameFormat format,
    size_t channelCount,
    size_t frameSampleCount,
    const string& filename,
    bool looping) :
    AudioInput(format, channelCount, frameSampleCount), m_looping(looping)
{
    auto fileStream = make_unique<ifstream>(filename, ifstream::binary);

    fileStream->seekg(0, fileStream->end);
    m_fileSize = static_cast<size_t>(fileStream->tellg());
    fileStream->seekg(0, fileStream->beg);

    if (m_fileSize % m_frame.size() != 0)
    {
        THROW_NOT_SUPPORTED_EXCEPTION("The file size must be a multiple of the frame size.");
    }

    m_fileStream = move(fileStream);
}

RawFileAudioInput::~RawFileAudioInput()
{
}

const PcmAudioFrame& RawFileAudioInput::read()
{
    if (m_fileStream->tellg() >= m_fileSize && m_looping)
    {
        m_fileStream->seekg(0, m_fileStream->beg);
    }

    *m_fileStream >> m_frame;

    return m_frame;
}

bool RawFileAudioInput::hasNext()
{
    return m_looping || m_fileStream->tellg() < m_fileSize;
}
