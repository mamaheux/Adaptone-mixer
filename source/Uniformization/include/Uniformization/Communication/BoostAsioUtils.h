#ifndef UNIFORMIZATION_COMMUNICATION_BOOST_ASIO_UTILS_DISCOVERER_H
#define UNIFORMIZATION_COMMUNICATION_BOOST_ASIO_UTILS_DISCOVERER_H

#include <Utils/Network/NetworkBuffer.h>

#include <boost/asio.hpp>

namespace adaptone
{
    constexpr std::size_t MaxNetworkBufferSize = 65535;

#if defined(_WIN32) || defined(_WIN64)
    typedef __socklen_t socklen_t;

    template<class Protocol, class DatagramSocketService>
    bool setReceivingTimeout(boost::asio::basic_socket<Protocol, DatagramSocketService>& socket, int timeoutMs)
    {
        DWORD timeoutValue = timeoutMs;
        return ::setsockopt(socket.native_handle(), SOL_SOCKET, SO_RCVTIMEO, &timeoutValue, sizeof(DWORD)) >= 0;
    }

#elif defined(__unix__) || defined(__linux__)

    template<class Protocol, class DatagramSocketService>
    bool setReceivingTimeout(boost::asio::basic_socket<Protocol, DatagramSocketService>& socket, int timeoutMs)
    {
        struct timeval tv;
        tv.tv_sec = timeoutMs / 1000;
        tv.tv_usec = (timeoutMs % 1000) * 1000;

        return ::setsockopt(socket.native_handle(), SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) >= 0;
    }

#endif

    inline std::size_t udpSocketReceiveTimeout(boost::asio::ip::udp::socket& socket,
        const boost::asio::mutable_buffer& buffer,
        boost::asio::ip::udp::endpoint& senderEndpoint,
        boost::asio::socket_base::message_flags flags,
        boost::system::error_code& ec)
    {
        socklen_t addr_len = senderEndpoint.capacity();

        int bytesReceived = ::recvfrom(socket.native_handle(),
            boost::asio::detail::buffer_cast_helper(buffer),
            boost::asio::detail::buffer_size_helper(buffer), flags,
            senderEndpoint.data(), &addr_len);

        if (bytesReceived < 0)
        {
            ec = boost::asio::error::fault;
        }

        return bytesReceived;
    }

    inline boost::asio::mutable_buffers_1 toAsioBuffer(NetworkBufferView buffer)
    {
        return boost::asio::buffer(buffer.data(), buffer.size());
    }

    inline boost::asio::mutable_buffers_1 toAsioBuffer(NetworkBufferView buffer, size_t size)
    {
        return boost::asio::buffer(buffer.data(), size);
    }
}

#endif
