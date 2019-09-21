#include <Uniformization/Communication/Messages/Tcp/TcpMessageReader.h>

#include <Uniformization/Communication/Messages/Tcp/FftRequestMessage.h>
#include <Uniformization/Communication/Messages/Tcp/FftResponseMessage.h>
#include <Uniformization/Communication/Messages/Tcp/HeartbeatMessage.h>
#include <Uniformization/Communication/Messages/Tcp/ProbeInitializationRequestMessage.h>
#include <Uniformization/Communication/Messages/Tcp/ProbeInitializationResponseMessage.h>
#include <Uniformization/Communication/Messages/Tcp/RecordRequestMessage.h>
#include <Uniformization/Communication/Messages/Tcp/RecordResponseMessage.h>

using namespace adaptone;
using namespace std;

#define ADD_HANDLE_FUNCTION(type) m_handlersById[type::Id] = [&](size_t messageSize, \
        function<void(const ProbeMessage&)>& readCallback) \
    { \
        readCallback(type::fromBuffer(m_buffer, messageSize)); \
    }

TcpMessageReader::TcpMessageReader() : m_buffer(MaxTcpMessageSize)
{
    ADD_HANDLE_FUNCTION(FftRequestMessage);
    ADD_HANDLE_FUNCTION(FftResponseMessage);
    ADD_HANDLE_FUNCTION(HeartbeatMessage);
    ADD_HANDLE_FUNCTION(ProbeInitializationRequestMessage);
    ADD_HANDLE_FUNCTION(ProbeInitializationResponseMessage);
    ADD_HANDLE_FUNCTION(RecordRequestMessage);
    ADD_HANDLE_FUNCTION(RecordResponseMessage);
}

TcpMessageReader::~TcpMessageReader()
{
}

void TcpMessageReader::read(boost::asio::ip::tcp::socket& socket, function<void(const ProbeMessage&)> readCallback)
{
    size_t messageSize = readTcpMessageData(socket, m_buffer);

    uint32_t id = boost::endian::big_to_native(*reinterpret_cast<uint32_t*>(m_buffer.data()));
    auto it = m_handlersById.find(id);
    if (it == m_handlersById.end())
    {
        THROW_NETWORK_EXCEPTION("Invalid id");
    }

    it->second(messageSize, readCallback);
}

size_t adaptone::readTcpMessageData(boost::asio::ip::tcp::socket& socket, NetworkBufferView buffer)
{
    size_t readDataSize = readTcpData(socket, buffer, sizeof(uint32_t));
    if (readDataSize != sizeof(uint32_t))
    {
        THROW_NETWORK_EXCEPTION("Invalid id");
    }

    uint32_t messageId = boost::endian::big_to_native(*reinterpret_cast<uint32_t*>(buffer.data()));
    size_t messageSize = sizeof(uint32_t);

    if (ProbeMessage::hasPayload(messageId))
    {
        readDataSize = readTcpData(socket, buffer.view(messageSize), sizeof(uint32_t));
        if (readDataSize != sizeof(uint32_t))
        {
            THROW_NETWORK_EXCEPTION("Invalid payload size");
        }

        size_t payloadSize = boost::endian::big_to_native(*reinterpret_cast<uint32_t*>(buffer.data()));
        messageSize += sizeof(uint32_t);

        if (payloadSize > buffer.size() - 2 * sizeof(uint32_t))
        {
            THROW_NETWORK_EXCEPTION("Too small buffer");
        }

        size_t receivedPayloadSize = 0;
        while (receivedPayloadSize < payloadSize)
        {
            size_t receivedSize =
                readTcpData(socket, buffer.view(messageSize), payloadSize - receivedPayloadSize);
            messageSize += receivedSize;
            receivedPayloadSize += receivedSize;
        }
    }
}
