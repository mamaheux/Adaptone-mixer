#ifndef UNIFORMIZATION_COMMUNICATION_MESSAGES_UDP_UDP_MESSAGE_READER_H
#define UNIFORMIZATION_COMMUNICATION_MESSAGES_UDP_UDP_MESSAGE_READER_H

#include <Uniformization/Communication/BoostAsioUtils.h>
#include <Uniformization/Communication/Messages/ProbeMessage.h>

#include <Utils/Exception/NetworkException.h>
#include <Utils/Network/NetworkBuffer.h>

#include <boost/asio.hpp>

#include <optional>
#include <functional>
#include <unordered_map>

namespace adaptone
{
    class UdpMessageReader
    {
        NetworkBuffer m_buffer;
        std::unordered_map<uint32_t, std::function<void(std::size_t,
            const boost::asio::ip::address&,
            std::function<void(const ProbeMessage&, const boost::asio::ip::address&)>&)>> m_handlersById;

    public:
        UdpMessageReader();
        virtual ~UdpMessageReader();

        void read(boost::asio::ip::udp::socket& socket,
            std::function<void(const ProbeMessage&, const boost::asio::ip::address&)> readCallback);
    };

    template<class T>
    T readUdpMessage(boost::asio::ip::udp::socket& socket, boost::asio::ip::udp::endpoint& endpoint,
        NetworkBufferView buffer)
    {
        boost::system::error_code error;
        std::size_t messageSize = udpSocketReceiveTimeout(socket, toAsioBuffer(buffer), endpoint, 0, error);
        if (error)
        {
            THROW_NETWORK_EXCEPTION("Unable to read the socket");
        }
        if (messageSize == 0)
        {
            THROW_NETWORK_EXCEPTION("Timeout");
        }

        return T::fromBuffer(buffer, messageSize);
    }

    template<class T>
    std::optional<T> readUdpMessageOrNull(boost::asio::ip::udp::socket& socket,
        boost::asio::ip::udp::endpoint& endpoint, NetworkBufferView buffer)
    {
        try
        {
            return std::optional<T>(readUdpMessage<T>(socket, endpoint, buffer));
        }
        catch (...)
        {
            return std::nullopt;
        }
    }
}

#endif
