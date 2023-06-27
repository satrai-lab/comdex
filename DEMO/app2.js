

    function stringToColor(str) {
        var hash = 0;
        for (var i = 0; i < str.length; i++) {
            hash = str.charCodeAt(i) + ((hash << 5) - hash);
        }
        var color = '';
        for (var i = 0; i < 3; i++) {
            var value = (hash >> (i * 8)) & 0xFF;
            color += ('00' + value.toString(16)).substr(-2);
        }
        return '#' + color;
    }


const socket = new WebSocket('ws://localhost:8080');
const terminal = document.getElementById('terminal');
const form = document.getElementById('form');
const input = document.getElementById('input');
const terminal2= document.getElementById('terminal2');

// When the WebSocket is open, echo a message to indicate this
socket.addEventListener('open', function (event) {
    terminal.value += '\nConnected to server\n';
});

// Log messages from the server
socket.addEventListener('message', function (event) {
    terminal.value += '\n' + event.data;
});

// Send a message to the server when the form is submitted
form.addEventListener('submit', function(event) {
  event.preventDefault();
        
        const message = {
            type: "message",
            payload: input.value
        };
        socket.send(JSON.stringify(message));
        input.value = '';
});

document.getElementById('commandForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const fileInput = document.getElementById('file');
    const file = fileInput.files.length > 0 ? fileInput.files[0] : null;
    const fileName = file ? file.name : ""; // Extract the file name if file exists, otherwise empty string

    const message = {
        type: "command",
        payload: {
            command: document.getElementById('command').value,
            file: fileName, // Use the file name instead of full path, or empty string
            broker_address: document.getElementById('broker_address').value,
            port: document.getElementById('port').value,
            qos: document.getElementById('qos').value,
            HLink: document.getElementById('HLink').value,
            singleidadvertisement: document.getElementById('singleidadvertisement').value,
        }
    };

    terminal2.value = "python3 actionhandler.py";

    // Appending the values to terminal2.value
    for(let key in message.payload) {
        let value = message.payload[key];
        // Check if the value is not empty
        if(value !== "" && value !== " ") {
            terminal2.value += " --" + key + " " + value;
        }
    }

    socket.send(JSON.stringify(message));
});


function setZoom() {
    // document.body.style.zoom = "79%"; // Adjust the zoom level as desired

  }

document.getElementById('file').addEventListener('change', function(event) {
    event.preventDefault();  // prevent the form from being submitted

    var fileInput = document.getElementById('file');
    var file = fileInput.files[0];

    if (file) {
        var reader = new FileReader();

        reader.onload = function() {
            try {
                var json = JSON.parse(reader.result);
                var output = JSON.stringify(json, null, 2);
                showModal(output);
            } catch (e) {
                alert("Invalid JSON file");
            }
        };

        reader.readAsText(file);
    }
  });

  function showModal(content) {
    var modal = document.createElement('div');
    modal.className = 'modal';

    var modalContent = document.createElement('div');
    modalContent.className = 'modal-content';

    var preElement = document.createElement('pre');
    preElement.textContent = content;

    modalContent.appendChild(preElement);
    modal.appendChild(modalContent);
    document.body.appendChild(modal);

    modal.style.display = 'block';

    modal.addEventListener('click', function(event) {
      if (event.target === modal) {
        modal.style.display = 'none';
        document.body.removeChild(modal);
      }
    });

    // Apply syntax highlighting to the JSON content
    hljs.highlightBlock(preElement);
  }

  document.getElementById('CreateFederation').addEventListener('click', function(event) {
    event.preventDefault();
        
    const message = {
        type: "Brokers"
    };
    
    socket.send(JSON.stringify(message));
});
