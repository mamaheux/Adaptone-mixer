#ifndef COMMUNICATION_APPLICATION_WEB_SOCKET_H
#define COMMUNICATION_APPLICATION_WEB_SOCKET_H

#include <Communication/Handlers/ConnectionHandler.h>
#include <Communication/Handlers/ApplicationMessageHandler.h>

#include <Utils/Logger/Logger.h>

#include <server_ws.hpp>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace adaptone
{
    class ApplicationWebSocket
    {
        std::shared_ptr<Logger> m_logger;
        std::shared_ptr<ConnectionHandler> m_connectionHandler;
        std::shared_ptr<ApplicationMessageHandler> m_applicationMessageHandler;

        SimpleWeb::SocketServer<SimpleWeb::WS> m_server;
        SimpleWeb::SocketServer<SimpleWeb::WS>::Endpoint& m_endpoint;
        std::shared_ptr<SimpleWeb::SocketServer<SimpleWeb::WS>::Connection> m_applicationConnection;

    public:
        ApplicationWebSocket(std::shared_ptr<Logger> logger,
            std::shared_ptr<ConnectionHandler> connectionHandler,
            std::shared_ptr<ApplicationMessageHandler> applicationMessageHandler,
            const std::string& endpoint,
            uint16_t port);
        virtual ~ApplicationWebSocket();

        void start();
        void stop();

        template<class T>
        void send(const T& object);

    private:
        SimpleWeb::StatusCode onHandshake(std::shared_ptr<SimpleWeb::SocketServer<SimpleWeb::WS>::Connection>
        connection);
        void onOpen(std::shared_ptr<SimpleWeb::SocketServer<SimpleWeb::WS>::Connection> connection);
        void onClose(std::shared_ptr<SimpleWeb::SocketServer<SimpleWeb::WS>::Connection> connection, int status,
            const std::string& reason);
        void onError(std::shared_ptr<SimpleWeb::SocketServer<SimpleWeb::WS>::Connection> connection,
            const SimpleWeb::error_code& ec);

        void onMessage(std::shared_ptr<SimpleWeb::SocketServer<SimpleWeb::WS>::Connection> connection,
            std::shared_ptr<SimpleWeb::SocketServer<SimpleWeb::WS>::InMessage> message);
    };

    template<class T>
    void ApplicationWebSocket::send(const T& object)
    {
        if (m_applicationConnection)
        {
            nlohmann::json j = object;
            m_applicationConnection->send(j.dump());
        }
    }
}

#endif
