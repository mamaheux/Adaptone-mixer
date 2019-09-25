#include <Uniformization/Communication/Messages/Tcp/FftResponseMessage.h>

using namespace adaptone;
using namespace std;

constexpr uint32_t FftResponseMessage::Id;
constexpr size_t FftResponseMessage::MinimumMessageSize;

static constexpr size_t FftIdFromBufferOffset = 8;

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
    verifyId(buffer, Id);
    verifyMessageSizeAtLeast(messageSize, MinimumMessageSize);

    uint16_t fftId = boost::endian::big_to_native(*reinterpret_cast<uint16_t*>(buffer.data() + FftIdFromBufferOffset));
    return FftResponseMessage(fftId,
        reinterpret_cast<complex<float>*>(buffer.data() + MinimumMessageSize),
        (messageSize - MinimumMessageSize) / sizeof(complex<float>));
}

void FftResponseMessage::serializePayload(NetworkBufferView buffer) const
{
    *reinterpret_cast<uint16_t*>(buffer.data()) = boost::endian::native_to_big(m_fftId);
    memcpy(buffer.data() + sizeof(m_fftId), m_fftValues, sizeof(complex<float>) * m_fftValueCount);
}
