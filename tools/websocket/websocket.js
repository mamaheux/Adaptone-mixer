socket = null;

document.getElementById("start-button").onclick = function() {
    try {
        socket = new WebSocket("ws://localhost:8080/mixer");

        socket.onerror = function(error) {
            console.error(error);
            socket = null;
        };
    
        socket.onopen = function(event) {
            console.log("Connexion établie.");
    
            this.onclose = function(event) {
                console.log("Connexion terminé.");
                socket = null;
            };
        
            this.onmessage = function(event) {
                console.log("Message:", event.data);
            };
        };
    } catch (ex) {
        console.error(ex);
    }    
};

document.getElementById("stop-button").onclick = function() {
    if (socket != null)
    {
        console.log("Connexion terminé.");
        socket.close();
        socket = null;
    }
    else
    {
        console.error("Connexion déjà terminé.");
    }
}

document.getElementById("send-button").onclick = function() {
    if (socket != null)
    {
        socket.send(document.getElementById("message-textarea").value);
    }
    else
    {
        console.error("Connexion terminé.");
    }
}