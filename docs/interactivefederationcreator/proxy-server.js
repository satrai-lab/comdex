const express = require('express');
const { spawn } = require('child_process');
const WebSocket = require('ws');

const app = express();
const port = 3000;

// Serve static files from the "public" directory
app.use(express.static('public'));

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});

// Setup WebSocket server
const wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', ws => {
  console.log('New client connected');

  ws.on('message', message => {
    console.log(`Received message: ${message}`);
    const python = spawn('python3', ['actionhandler.py', message]);
    
    python.stdout.on('data', (data) => {
      ws.send(data.toString());
    });

    python.stderr.on('data', (data) => {
      console.error(`stderr: ${data}`);
      ws.send(data.toString());
    });

    python.on('close', (code) => {
      console.log(`child process exited with code ${code}`);
      ws.send(data.toString());
    });
  });

  ws.on('close', () => {
    console.log('Client disconnected');
  });
});