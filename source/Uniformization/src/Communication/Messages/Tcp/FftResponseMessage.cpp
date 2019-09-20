#include <Uniformization/Communication/Messages/Tcp/FftResponseMessage.h>

using namespace adaptone;
using namespace std;

constexpr uint32_t FftResponseMessage::Id;

FftResponseMessage::FftResponseMessage(uint16_t fftId, const complex<float>* fftValues, size_t fftValueCount) :
    PayloadMessage(Id, sizeof(m_fftId) + sizeof(complex<float>) * fftValueCount),
    m_fftId(fftId),
    m_fftValues(fftValues),
    m_fftValueCount(fftValueCount)
{
}

FftResponseMessage::~FftResponseMessage()
{
}

void FftResponseMessage::serializePayload(NetworkBufferView& buffer)
{
    *reinterpret_cast<uint16_t*>(buffer.data()) = boost::endian::native_to_big(m_fftId);
    memcpy(buffer.data() + sizeof(m_fftId), m_fftValues, sizeof(complex<float>) * m_fftValueCount);
}
