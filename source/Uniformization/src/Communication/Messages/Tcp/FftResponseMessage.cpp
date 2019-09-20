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

FftResponseMessage FftResponseMessage::fromBuffer(NetworkBufferView buffer, size_t messageSize)
{
    constexpr size_t MinimumSize = 10;
    verifyId(buffer, Id);
    verifyMessageSizeAtLeast(messageSize, MinimumSize);

    return FftResponseMessage(boost::endian::big_to_native(*reinterpret_cast<uint16_t*>(buffer.data() + 8)),
        reinterpret_cast<complex<float>*>(buffer.data() + MinimumSize),
        (messageSize - MinimumSize) / sizeof(complex<float>));
}

void FftResponseMessage::serializePayload(NetworkBufferView buffer)
{
    *reinterpret_cast<uint16_t*>(buffer.data()) = boost::endian::native_to_big(m_fftId);
    memcpy(buffer.data() + sizeof(m_fftId), m_fftValues, sizeof(complex<float>) * m_fftValueCount);
}
