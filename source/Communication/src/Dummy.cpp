#include <Communication/Dummy.h>

#include <server_ws.hpp>

using namespace adaptone;
using namespace std;

using WsServer = SimpleWeb::SocketServer<SimpleWeb::WS>;

void adaptone::startWebSocket()
{
    WsServer server;
    server.config.port = 8080;

    auto& echo = server.endpoint["^/echo/?$"];

    echo.on_message = [](shared_ptr<WsServer::Connection> connection, shared_ptr<WsServer::InMessage> in_message)
    {
        auto out_message = in_message->string();

        cout << "Server: Message received: \"" << out_message << "\" from " << connection.get() << endl;

        cout << "Server: Sending message \"" << out_message << "\" to " << connection.get() << endl;

        connection->send(out_message, [](const SimpleWeb::error_code& ec)
        {
            if (ec)
            {
                cout << "Server: Error sending message. " <<
                    "Error: " << ec << ", error message: " << ec.message() << endl;
            }
        });
    };

    echo.on_open = [](shared_ptr<WsServer::Connection> connection)
    {
        cout << "Server: Opened connection " << connection.get() << endl;
    };

    echo.on_close = [](shared_ptr<WsServer::Connection> connection, int status, const string& /*reason*/)
    {
        cout << "Server: Closed connection " << connection.get() << " with status code " << status << endl;
    };

    echo.on_error = [](shared_ptr<WsServer::Connection> connection, const SimpleWeb::error_code& ec)
    {
        cout << "Server: Error in connection " << connection.get() << ". "
            << "Error: " << ec << ", error message: " << ec.message() << endl;
    };
}
