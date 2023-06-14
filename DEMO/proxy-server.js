const express = require('express');
const { spawn } = require('child_process');
const WebSocket = require('ws');
const path = require('path');

const app = express();
const port = 3000;

// Serve static files from the "public" directory
app.use(express.static('public'));

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});

// Setup WebSocket server
const wss = new WebSocket.Server({ port: 8080 });

const fs = require('fs');
const childProcess = require('child_process');

let childProcesses = [];

wss.on('connection', ws => {
  console.log('New client connected');

  ws.on('message', message => {
    console.log(`Received message: ${message}`);
    let parsedMessage = JSON.parse(message);
    if (parsedMessage.type == "Brokers") {
      fs.readdir('./', (err, files) => {
        if (err) {
          console.log(`Error reading directory: ${err}`);
          ws.send(`Error reading directory: ${err}`);
        } else {
          let confFiles = files.filter(file => file.endsWith('.conf'));

          confFiles.forEach(confFile => {
            const newProcess = childProcess.spawn('mosquitto', ['-c', confFile]);
            childProcesses.push(newProcess);

            newProcess.stdout.on('data', (data) => {
              ws.send(data.toString());
            });

            newProcess.stderr.on('data', (data) => {
              console.error(`stderr: ${data}`);
              ws.send(data.toString());
            });

            newProcess.on('close', (code) => {
              console.log(`child process exited with code ${code}`);
              ws.send(code.toString());
            });
          });
        }
      });
    } else if (parsedMessage.type == "command") {

      parsedMessage = parsedMessage.payload
      let args = [];
      if (parsedMessage.command.trim() !== '') {
        args.push('-c', parsedMessage.command);
      }
      if (parsedMessage.file.trim() !== '') {
        args.push('-f', parsedMessage.file);
      }
      if (parsedMessage.broker_address.trim() !== '') {
        args.push('-b', parsedMessage.broker_address);
      }
      if (parsedMessage.port.trim() !== '') {
        args.push('-p', parsedMessage.port);
      }
      if (parsedMessage.qos.trim() !== '') {
        args.push('-q', parsedMessage.qos);
      }
      if (parsedMessage.HLink.trim() !== '') {
        args.push('-H', parsedMessage.HLink);
      }
      if (parsedMessage.singleidadvertisement.trim() !== '') {
        args.push('-A', parsedMessage.singleidadvertisement);
      }

      const python = childProcess.spawn('python3', ['actionhandler.py'].concat(args));
      childProcesses.push(python);

      python.stdout.on('data', (data) => {
        ws.send(data.toString());
      });

      python.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
        ws.send(data.toString());
      });

      python.on('close', (code) => {
        console.log(`child process exited with code ${code}`);
        ws.send(code.toString());
      });
    }
  });
});

process.on('SIGINT', () => {
  console.log('Stopping server and killing child processes...');

  // Send SIGINT signal to each child process
  childProcesses.forEach((childProcess) => {
    childProcess.kill('SIGINT');
  });

  // Delete .conf files in the current folder
  const currentFolder = process.cwd();
  const files = fs.readdirSync(currentFolder);
  files.forEach((file) => {
    if (file.endsWith('.conf')) {
      const filePath = path.join(currentFolder, file);
      fs.unlinkSync(filePath);
      console.log(`Deleted file: ${filePath}`);
    }
  });

  process.exit(); // Exit the server process
});