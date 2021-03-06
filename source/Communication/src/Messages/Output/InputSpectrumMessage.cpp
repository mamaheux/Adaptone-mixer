#include <Communication/Messages/Output/InputSpectrumMessage.h>

using namespace adaptone;
using namespace std;

ChannelSpectrum::ChannelSpectrum() : m_channelId(0), m_points()
{
}

ChannelSpectrum::ChannelSpectrum(size_t channelId, vector<SpectrumPoint> points) :
    m_channelId(channelId), m_points(move(points))
{
}

ChannelSpectrum::~ChannelSpectrum()
{
}

constexpr size_t InputSpectrumMessage::SeqId;

InputSpectrumMessage::InputSpectrumMessage() : ApplicationMessage(SeqId), m_channelSpectrums()
{
}

InputSpectrumMessage::InputSpectrumMessage(vector<ChannelSpectrum> channelSpectrums) :
    ApplicationMessage(SeqId), m_channelSpectrums(move(channelSpectrums))
{
}

InputSpectrumMessage::~InputSpectrumMessage()
{
}
