#include "server_wss.hpp"

using namespace std;

using WssServer = SimpleWeb::SocketServer<SimpleWeb::WSS>;

int main() {
  // Secure socket sur le port 8080
  // Écoute sur le port 8080 et echo tous les mesages de façon asynchrone
  WssServer server("server.crt", "server.key");
  server.config.port = 8080;
  auto &echo = server.endpoint["^/echo/?$"];

  echo.on_message = [](shared_ptr<WssServer::Connection> connection, shared_ptr<WssServer::InMessage> in_message) {
    auto out_message = in_message->string();

    cout << "Server: Message received: \"" << out_message << "\" from " << connection.get() << endl;

    cout << "Server: Sending message \"" << out_message << "\" to " << connection.get() << endl;

    connection->send(out_message, [](const SimpleWeb::error_code &ec) {
      if(ec) {
        cout << "Server: Error sending message. " <<
            "Error: " << ec << ", error message: " << ec.message() << endl;
      }
    });
  };

  echo.on_open = [](shared_ptr<WssServer::Connection> connection) {
    cout << "Server: Opened connection " << connection.get() << endl;
  };

  echo.on_close = [](shared_ptr<WssServer::Connection> connection, int status, const string &) {
    cout << "Server: Closed connection " << connection.get() << " with status code " << status << endl;
  };

  echo.on_error = [](shared_ptr<WssServer::Connection> connection, const SimpleWeb::error_code &ec) {
    cout << "Server: Error in connection " << connection.get() << ". "
         << "Error: " << ec << ", error message: " << ec.message() << endl;
  };
}
