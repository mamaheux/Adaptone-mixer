#ifndef UNIFORMIZATION_COMMUNICATION_MESSAGES_TCP_TCP_MESSAGE_HANDLER_H
#define UNIFORMIZATION_COMMUNICATION_MESSAGES_TCP_TCP_MESSAGE_HANDLER_H

#include <Uniformization/Communication/BoostAsioUtils.h>
#include <Uniformization/Communication/Messages/ProbeMessage.h>

#include <Utils/Exception/NetworkException.h>
#include <Utils/Network/NetworkBuffer.h>

#include <boost/asio.hpp>
#include <boost/endian/arithmetic.hpp>

#include <optional>

namespace adaptone
{
    constexpr std::size_t MaxTcpMessageSize = 10485760;

    class TcpMessageReader
    {
        NetworkBuffer m_buffer;
        std::unordered_map<uint32_t, std::function<void(std::size_t,
            std::function<void(const ProbeMessage&)>&)>> m_handlersById;

    public:
        TcpMessageReader();
        virtual ~TcpMessageReader();

        void read(boost::asio::ip::tcp::socket& socket, std::function<void(const ProbeMessage&)> readCallback);
    };

    inline std::size_t readTcpData(boost::asio::ip::tcp::socket& socket, NetworkBufferView buffer, std::size_t dataSize)
    {
        boost::system::error_code error;
        std::size_t messageSize = tcpSocketReceiveTimeout(socket, toAsioBuffer(buffer, dataSize), 0, error);
        if (error)
        {
            THROW_NETWORK_EXCEPTION("Unable to read the socket");
        }
        if (messageSize == 0)
        {
            THROW_NETWORK_EXCEPTION("Timeout");
        }

        return messageSize;
    }

    std::size_t readTcpMessageData(boost::asio::ip::tcp::socket& socket, NetworkBufferView buffer);

    template<class T>
    T readTcpMessage(boost::asio::ip::tcp::socket& socket, NetworkBufferView buffer)
    {
        std::size_t messageSize = readTcpMessageData(socket, buffer);
        return T::fromBuffer(buffer, messageSize);
    }

    template<class Response, class Request>
    Response sendRequest(boost::asio::ip::tcp::socket& socket, NetworkBufferView buffer, const Request& request)
    {
        boost::system::error_code error;

        request.toBuffer(buffer);
        socket.send(toAsioBuffer(buffer, request.fullSize()), 0, error);
        if (error)
        {
            THROW_NETWORK_EXCEPTION("Unable to send request");
        }

        return readTcpMessage<Response>(socket, buffer);
    }
}

#endif
