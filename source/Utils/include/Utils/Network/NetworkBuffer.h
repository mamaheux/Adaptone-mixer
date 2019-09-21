#ifndef UTILS_NETWORK_NETWORK_BUFFER_H
#define UTILS_NETWORK_NETWORK_BUFFER_H

#include <Utils/ClassMacro.h>

#include <cstddef>
#include <cstdint>

namespace adaptone
{
    class NetworkBuffer;

    class NetworkBufferView
    {
        uint8_t* m_data;
        std::size_t m_size;

    public:
        NetworkBufferView(uint8_t* data, std::size_t size);
        NetworkBufferView(NetworkBuffer& buffer);
        virtual ~NetworkBufferView();

        uint8_t* data();
        const uint8_t* data() const;
        std::size_t size() const;

        NetworkBufferView view(std::size_t offset = 0);
    };

    inline uint8_t* NetworkBufferView::data()
    {
        return m_data;
    }

    inline const uint8_t* NetworkBufferView::data() const
    {
        return m_data;
    }

    inline std::size_t NetworkBufferView::size() const
    {
        return m_size;
    }

    inline NetworkBufferView NetworkBufferView::view(std::size_t offset)
    {
        return NetworkBufferView(m_data + offset, m_size - offset);
    }

    class NetworkBuffer
    {
        uint8_t* m_data;
        std::size_t m_size;
    public:
        NetworkBuffer(std::size_t size);
        virtual ~NetworkBuffer();

        DECLARE_NOT_COPYABLE(NetworkBuffer);
        DECLARE_NOT_MOVABLE(NetworkBuffer);

        uint8_t* data();
        const uint8_t* data() const;
        std::size_t size() const;

        NetworkBufferView view(std::size_t offset = 0);
    };

    inline uint8_t* NetworkBuffer::data()
    {
        return m_data;
    }

    inline const uint8_t* NetworkBuffer::data() const
    {
        return m_data;
    }

    inline std::size_t NetworkBuffer::size() const
    {
        return m_size;
    }

    inline NetworkBufferView NetworkBuffer::view(std::size_t offset)
    {
        return NetworkBufferView(m_data + offset, m_size - offset);
    }
}

#endif
