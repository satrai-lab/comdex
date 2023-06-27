document.addEventListener("DOMContentLoaded", function(){
    var cy = window.cy = cytoscape({
        container: document.getElementById('cy'),
    
        boxSelectionEnabled: true, // Allow box selection
        autounselectify: false, // Allow elements to be selected
    

        style: [
            {
                selector: 'node',
                style: {
                    'content': 'data(id)',
                    'background-color': 'data(communityColor)'
                }
            },
            {
                selector: 'edge',
                style: {
                    'curve-style': 'bezier',
                    'control-point-step-size': 40,
                    'target-arrow-shape': 'triangle',
                    'label': 'data(topic)',
                    'text-rotation': 'autorotate',
                    'text-margin-y': -10
                }
            },
            {
                selector: 'node:selected',
                style: {
                    'border-color': 'black',
                    'border-width': '3px'
                }
            },
            {
                selector: 'edge:selected',
                style: {
                    'line-color': 'black',
                    'width': '4px'
                }
            }

        ],

        elements: {
            nodes: [],
            edges: []
        }
    });

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
    document.getElementById('deleteButton').addEventListener('click', function() {
        var selectedNodes = cy.nodes(':selected');
        var selectedEdges = cy.edges(':selected');
        // Update dropdown options
        selectedNodes.forEach(function(node) {
            var option = document.querySelector(`option[value="${node.id()}"]`);
            if (option) option.remove();
        });
        cy.remove(selectedNodes);
        cy.remove(selectedEdges);
    });

    cy.on('remove', 'node', function() {
        var nodes = cy.nodes().map(node => node.id());
        var sourceNodeSelect = document.getElementById('sourceNode');
        var targetNodeSelect = document.getElementById('targetNode');
        
        sourceNodeSelect.innerHTML = '';
        targetNodeSelect.innerHTML = '';

        for (var i = 0; i < nodes.length; i++) {
            var option = document.createElement('option');
            option.text = nodes[i];
            sourceNodeSelect.add(option);
            targetNodeSelect.add(option.cloneNode(true));
        }
    });
    document.getElementById('addNodeForm').addEventListener('submit', function(event) {
        event.preventDefault();

        const id = document.getElementById('nodeName').value;
        const ip = document.getElementById('nodeIp').value;
        const port = document.getElementById('nodePort').value;
        const community = document.getElementById('nodeCommunity').value;
        const communityColor = stringToColor(community);

        cy.add({
            group: 'nodes',
            data: { id, ip, port, community, communityColor },
            position: { x: 200, y: 200 }
        });

        const option = document.createElement('option');
        option.text = id;
        document.getElementById('sourceNode').add(option);
        document.getElementById('targetNode').add(option.cloneNode(true));
    });

    document.getElementById('addEdgeForm').addEventListener('submit', function(event) {
        event.preventDefault();

        const source = document.getElementById('sourceNode').value;
        const target = document.getElementById('targetNode').value;
        const topic = document.getElementById('edgeTopic').value;

        cy.add({
            group: 'edges',
            data: { id: 'edge' + source + '_' + target, source, target, topic }
        });
    });

    function hasCycle(node, visited, recStack) {
        if (!visited[node.id()]) {
            visited[node.id()] = true;
            recStack[node.id()] = true;
            var neighbors = node.outgoers().nodes();
            for (var i = 0; i < neighbors.length; i++) {
                if (!visited[neighbors[i].id()] && hasCycle(neighbors[i], visited, recStack)) {
                    return true;
                } else if (recStack[neighbors[i].id()]) {
                    return true;
                }
            }
        }
        recStack[node.id()] = false;  // remove the node from recursion stack
        return false;
    }
    
    function hasCycle(node, visited, recStack) {
    if (!visited[node.id()]) {
        visited[node.id()] = true;
        recStack[node.id()] = true;
        var neighbors = node.outgoers().nodes();
        for (var i = 0; i < neighbors.length; i++) {
            if (!visited[neighbors[i].id()] && hasCycle(neighbors[i], visited, recStack)) {
                return true;
            } else if (recStack[neighbors[i].id()]) {
                return true;
            }
        }
    }
    recStack[node.id()] = false;  // remove the node from recursion stack
    return false;
}
document.getElementById('exportConf').addEventListener('click', function() {
    // check for cycles
    var nodes = cy.nodes();
    var visited = {};
    var recStack = {};
    for (var i = 0; i < nodes.length; i++) {
        if (hasCycle(nodes[i], visited, recStack)) {
            alert("Error: Cycle detected in the network. Please correct the structure and try again.");
            return;
        }
    }

    var communities = {};
    cy.nodes().forEach(function(node) {
        var data = node.data();
        var community = data.community;
        if (!communities[community]) {
            communities[community] = [];
        }
        communities[community].push(data);
    });

    for (var community in communities) {
        var nodes = communities[community];
        for (var i = 0; i < nodes.length; i++) {
            var node = nodes[i];
            var conf = [];
            // Add additional configurations
            conf.push("pid_file /var/run/mosquitto/mosquitto.pid");
            conf.push("max_queued_messages 4000");
            conf.push("persistence true");
            conf.push("persistence_location .");
            conf.push("log_dest stdout");
            conf.push(`listener ${node.port} ${node.ip}`);
            conf.push('allow_anonymous true');

            var edges = cy.edges(`[source = "${node.id}"]`);
            edges.forEach(function(edge) {
                var target = edge.target();
                conf.push('');
                conf.push(`connection bridge_to_${target.id()}`);
                conf.push(`address ${target.data().ip}:${target.data().port}`);
                // Append 'provider/' prefix to topic and handle 'all' case
                var topic = edge.data().topic === 'all' ? 'provider/#' : `provider/${edge.data().topic}/#`;
                conf.push(`topic ${topic} out 2 "" ""`);
            });

            var confStr = conf.join('\n');
            var blob = new Blob([confStr], {type: 'text/plain'});
            var url = URL.createObjectURL(blob);
            var link = document.createElement('a');
            link.href = url;
            link.download = `${community}_${node.id}.conf`;
            link.style.display = 'none';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }
});

    cy.on('tap', 'node', function(){
        try { 
            window.cyNodeEditing.enable(this);
        } catch(e){
            // window.alert('Enable node editing by clicking the node');
        }
    });

    cy.edgehandles({});
});

const socket = new WebSocket('ws://localhost:8080');
const terminal = document.getElementById('terminal');
const form = document.getElementById('form');
const input = document.getElementById('input');

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
  
  const message = input.value;
  socket.send(message);

  input.value = '';
});
