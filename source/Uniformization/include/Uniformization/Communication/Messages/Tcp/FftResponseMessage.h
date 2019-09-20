#ifndef UNIFORMIZATION_COMMUNICATION_MESSAGES_TCP_FFT_RESPONSE_MESSAGE_H
#define UNIFORMIZATION_COMMUNICATION_MESSAGES_TCP_FFT_RESPONSE_MESSAGE_H

#include <Uniformization/Communication/Messages/PayloadMessage.h>

#include <complex>

namespace adaptone
{
    /**
     * The fft values are not copied. So, the message life time depends on the network buffer life time.
     */
    class FftResponseMessage : public PayloadMessage
    {
    public:
        static constexpr uint32_t Id = 8;

    private:
        uint16_t m_fftId;
        const std::complex<float>* m_fftValues;
        std::size_t m_fftValueCount;

    public:
        FftResponseMessage(uint16_t fftId, const std::complex<float>* fftValues, std::size_t fftValueCount);
        ~FftResponseMessage() override;

        uint8_t fftId() const;
        const std::complex<float>* fftValues() const;
        std::size_t fftValueCount() const;

    protected:
        void serializePayload(NetworkBufferView& buffer) override;
    };

    inline uint8_t FftResponseMessage::fftId() const
    {
        return m_fftId;
    }

    inline const std::complex<float>* FftResponseMessage::fftValues() const
    {
        return m_fftValues;
    }

    inline std::size_t FftResponseMessage::fftValueCount() const
    {
        return m_fftValueCount;
    }
}

#endif
