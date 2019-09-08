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
                console.log("Connexion terminée.");
                socket = null;
            };
        
            this.onmessage = function(event) {
                var message = JSON.parse(event.data);
                if (message.seqId == 21)
                {
                    document.getElementById("input0Level").innerHTML = (20 * Math.log10(message.data.inputAfterGain[0])).toString();
                    document.getElementById("input1Level").innerHTML = (20 * Math.log10(message.data.inputAfterGain[1])).toString();
                }
                else
                {
                    console.log(message);
                }
            };

            socket.send("{\"seqId\": 11, \"data\": {\"gains\": [3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] }}");
            socket.send("{\"seqId\": 13, \"data\": {\"channelId\": 0, \"gain\": 1.0 }}");
            socket.send("{\"seqId\": 13, \"data\": {\"channelId\": 1, \"gain\": 1.0 }}");
            socket.send("{\"seqId\": 17, \"data\": {\"gain\": 1.0}}");
            socket.send("{\"seqId\": 12, \"data\": {\"channelId\": 0, \"gains\": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] }}");
            socket.send("{\"seqId\": 12, \"data\": {\"channelId\": 1, \"gains\": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] }}");
            socket.send("{\"seqId\": 15, \"data\": {\"gains\": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}}");
        };
    } catch (ex) {
        console.error(ex);
    }    
};

document.getElementById("stop-button").onclick = function() {
    if (socket != null) {
        console.log("Connexion terminée.");
        socket.close();
        socket = null;
    } else {
        console.error("Connexion déjà terminée.");
    }
}

document.getElementById("send-button").onclick = function() {
    if (socket != null) {
        socket.send(document.getElementById("message-textarea").value);
    } else {
        console.error("Connexion terminé.");
    }
}

