#ifndef UNIFORMIZATION_COMMUNICATION_MESSAGES_PROBE_MESSAGE_H
#define UNIFORMIZATION_COMMUNICATION_MESSAGES_PROBE_MESSAGE_H

#include <Uniformization/Communication/NetworkBuffer.h>

#include <Utils/Exception/InvalidValueException.h>

#include <boost/endian/arithmetic.hpp>

#include <cstddef>

namespace adaptone
{
    class ProbeMessage
    {
        uint32_t m_id;
        std::size_t m_fullSize;

    public:
        ProbeMessage(uint32_t id, std::size_t payloadSize);
        virtual ~ProbeMessage();

        uint32_t id();
        std::size_t fullSize();

        void toBuffer(NetworkBuffer& buffer);

    protected:
        virtual void serialize(NetworkBufferView& buffer) = 0;
    };

    inline uint32_t ProbeMessage::id()
    {
        return m_id;
    }

    inline std::size_t ProbeMessage::fullSize()
    {
        return m_fullSize;
    }

    inline void ProbeMessage::toBuffer(NetworkBuffer& buffer)
    {
        if (buffer.size() < m_fullSize)
        {
            THROW_INVALID_VALUE_EXCEPTION("Network buffer size", "");
        }

        *reinterpret_cast<uint32_t*>(buffer.data()) = boost::endian::native_to_big(m_id);

        NetworkBufferView view = buffer.view(sizeof(m_id));
        serialize(view);
    }
}

#endif
