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
};

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
    
        const boundingBox = cy.elements().boundingBox();
        
        const posX = Math.floor(Math.random() * (boundingBox.x2 - boundingBox.x1) + boundingBox.x1);
        const posY = Math.floor(Math.random() * (boundingBox.y2 - boundingBox.y1) + boundingBox.y1);

    
        cy.add({
            group: 'nodes',
            data: { id, ip, port, community, communityColor },
            position: { x: posX, y: posY }  // Update position to be random
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
    
        // Create a unique edge id by appending a timestamp
        const uniqueEdgeId = 'edge' + source + '_' + target + '_' + Date.now();
    
        cy.add({
            group: 'edges',
            data: { id: uniqueEdgeId, source, target, topic }
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
            //conf.push("pid_file ./mosquitto.pid");
            conf.push("max_queued_messages 4000");
            conf.push("persistence true");
            // Define the persistence location for each node based on its name or id
            var persistenceLocation = `./${node.id}/`;
            conf.push(`persistence_location ${persistenceLocation}`);
            conf.push("log_dest stdout");
            conf.push(`listener ${node.port} ${node.ip}`);
            conf.push('allow_anonymous true');

            var edges = cy.edges(`[source = "${node.id}"]`);
            edges.forEach(function(edge) {
                var target = edge.target();
                conf.push('');
                let randomString = Math.random().toString(36).substring(7);
                conf.push(`connection ${node.id.replace(/\s+/g, '_')}${randomString}bridge_to_${target.id().replace(/\s+/g, '_')}${randomString}`);

                conf.push(`address ${target.data().ip}:${target.data().port}`);
                // Append 'provider/' prefix to topic and handle 'all' case
                var topic = edge.data().topic.toLowerCase() === 'all' ? 'provider/#' : `provider/+/+/+/+/${edge.data().topic}`;
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



document.getElementById('importTopology').addEventListener('change', function(event) {
    var fileInput = event.target;
    var file = fileInput.files[0];

    if (file) {
        var reader = new FileReader();

        reader.onload = function() {
            try {
                var json = JSON.parse(reader.result);

                // Clear the current graph
                cy.elements().remove();

                // Import nodes
                if (Array.isArray(json.nodes)) {
                    json.nodes.forEach(function(nodeData) {
                        if (nodeData.id && nodeData.position && typeof nodeData.position.x === 'number' && typeof nodeData.position.y === 'number') {
                            cy.add({
                                group: 'nodes',
                                data: {
                                    id: nodeData.id,
                                    ip: nodeData.ip || '',
                                    port: nodeData.port || '',
                                    community: nodeData.community || '',
                                    communityColor: stringToColor(nodeData.community || '')
                                },
                                position: {
                                    x: nodeData.position.x,
                                    y: nodeData.position.y
                                }
                            });
                        }
                    });
                }

                // Import edges
                if (Array.isArray(json.edges)) {
                    json.edges.forEach(function(edgeData) {
                        if (edgeData.id && edgeData.source && edgeData.target && edgeData.topic) {
                            cy.add({
                                group: 'edges',
                                data: {
                                    id: edgeData.id,
                                    source: edgeData.source,
                                    target: edgeData.target,
                                    topic: edgeData.topic
                                }
                            });
                        }
                    });
                }
            } catch (e) {
                console.log("Error parsing JSON file:", e);
                alert("Invalid JSON file");
            }
        };

        reader.readAsText(file);
    }
});





function setZoom() {
    // document.body.style.zoom = "79%"; // Adjust the zoom level as desired

  };

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
  };

