#include <Communication/ApplicationWebSocket.h>

using namespace adaptone;
using namespace std;
using namespace std::placeholders;

using WsServer = SimpleWeb::SocketServer<SimpleWeb::WS>;
using nlohmann::json;

ApplicationWebSocket::ApplicationWebSocket(shared_ptr<Logger> logger,
    shared_ptr<ConnectionHandler> connectionHandler,
    shared_ptr<ApplicationMessageHandler> applicationMessageHandler,
    const string& endpoint,
    uint16_t port) :
    m_logger(logger),
    m_connectionHandler(connectionHandler),
    m_applicationMessageHandler(applicationMessageHandler),

    m_server(),
    m_endpoint(m_server.endpoint[endpoint]),
    m_applicationConnection(nullptr)
{
    m_server.config.port = port;
    m_server.config.thread_pool_size = 1;

    m_endpoint.on_handshake = bind(&ApplicationWebSocket::onHandshake, this, _1);
    m_endpoint.on_open = bind(&ApplicationWebSocket::onOpen, this, _1);
    m_endpoint.on_close = bind(&ApplicationWebSocket::onClose, this, _1, _2, _3);
    m_endpoint.on_error = bind(&ApplicationWebSocket::onError, this, _1, _2);

    m_endpoint.on_message = bind(&ApplicationWebSocket::onMessage, this, _1, _2);
}

ApplicationWebSocket::~ApplicationWebSocket()
{
}

void ApplicationWebSocket::start()
{
    m_server.start();
}

void ApplicationWebSocket::stop()
{
    m_server.stop();
}

string connectionToString(shared_ptr<WsServer::Connection> connection)
{
    return "[" + connection->remote_endpoint_address() + "]:" + to_string(connection->remote_endpoint_port());
}

SimpleWeb::StatusCode ApplicationWebSocket::onHandshake(shared_ptr<WsServer::Connection> connection)
{
    if (!m_applicationConnection)
    {
        m_logger->log(Logger::Level::Information, "Connection accepted (" + connectionToString(connection) + ")");
        return SimpleWeb::StatusCode::success_ok;
    }

    m_logger->log(Logger::Level::Information, "Connection refused (" + connectionToString(connection) + ")");
    return SimpleWeb::StatusCode::client_error_conflict;
}

void ApplicationWebSocket::onOpen(shared_ptr<WsServer::Connection> connection)
{
    if (!m_applicationConnection)
    {
        m_logger->log(Logger::Level::Information, "Connection opened (" + connectionToString(connection) + ")");
        m_applicationConnection = connection;
        m_connectionHandler->handleConnection();
    }
}

void ApplicationWebSocket::onClose(shared_ptr<WsServer::Connection> connection, int status, const string& reason)
{
    if (m_applicationConnection == connection)
    {
        m_logger->log(Logger::Level::Information,
            "Connection closed (" + connectionToString(connection) + " (" + to_string(status) + ", " + reason + "))");
        m_applicationConnection.reset();
        m_connectionHandler->handleDisconnection();
    }
}

void ApplicationWebSocket::onError(shared_ptr<WsServer::Connection> connection, const SimpleWeb::error_code& ec)
{
    if (m_applicationConnection == connection)
    {
        m_logger->log(Logger::Level::Error,
            "Connection closed (" + connectionToString(connection) + " (" + ec.message() + ")");
        m_applicationConnection.reset();
        m_connectionHandler->handleDisconnection();
    }
}

void ApplicationWebSocket::onMessage(shared_ptr<WsServer::Connection> connection,
    shared_ptr<WsServer::InMessage> message)
{
    try
    {
        json j = json::parse(message->string());
        m_applicationMessageHandler->handle(j, [this](const ApplicationMessage& messageToSend)
        {
            send(messageToSend);
        });
    }
    catch (exception& ex)
    {
        m_logger->log(Logger::Level::Error, ex, "message=" + message->string());
    }
    catch (...)
    {
        m_logger->log(Logger::Level::Error, "Unknown error message=" + message->string());
    }
}
