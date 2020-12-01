#ifndef UNIFORMIZATION_COMMUNICATION_MESSAGES_PROBE_MESSAGE_H
#define UNIFORMIZATION_COMMUNICATION_MESSAGES_PROBE_MESSAGE_H

#include <Utils/Exception/InvalidValueException.h>
#include <Utils/Network/NetworkBuffer.h>

#include <boost/endian/conversion.hpp>

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

        uint32_t id() const;
        std::size_t fullSize() const;

        void toBuffer(NetworkBufferView buffer) const;

        static bool hasPayload(uint32_t id);

    protected:
        virtual void serialize(NetworkBufferView buffer) const = 0;

        static void verifyId(NetworkBufferView buffer, uint32_t id);
        static void verifyMessageSize(std::size_t messageSize, std::size_t validMessageSize);
        static void verifyMessageSizeAtLeast(std::size_t messageSize, std::size_t validMessageSize);
    };

    inline uint32_t ProbeMessage::id() const
    {
        return m_id;
    }

    inline std::size_t ProbeMessage::fullSize() const
    {
        return m_fullSize;
    }

    inline void ProbeMessage::toBuffer(NetworkBufferView buffer) const
    {
        if (buffer.size() < m_fullSize)
        {
            THROW_INVALID_VALUE_EXCEPTION("Network buffer size", "");
        }

        *reinterpret_cast<uint32_t*>(buffer.data()) = boost::endian::native_to_big(m_id);

        serialize(buffer.view(sizeof(m_id)));
    }

    inline void ProbeMessage::verifyId(NetworkBufferView buffer, uint32_t id)
    {
        if (boost::endian::big_to_native(*reinterpret_cast<uint32_t*>(buffer.data())) != id)
        {
            THROW_INVALID_VALUE_EXCEPTION("Message id", "");
        }
    }

    inline void ProbeMessage::verifyMessageSize(std::size_t messageSize, std::size_t validMessageSize)
    {
        if (messageSize != validMessageSize)
        {
            THROW_INVALID_VALUE_EXCEPTION("Message size", "");
        }
    }

    inline void ProbeMessage::verifyMessageSizeAtLeast(std::size_t messageSize, std::size_t minimumMessageSize)
    {
        if (messageSize < minimumMessageSize)
        {
            THROW_INVALID_VALUE_EXCEPTION("Message size", "");
        }
    }
}

#endif
