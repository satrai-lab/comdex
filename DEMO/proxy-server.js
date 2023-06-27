const express = require('express');
const { spawn } = require('child_process');
const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

const app = express();
const port = 3000;

// Serve static files from the "public" directory
app.use(express.static('public'));

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});

// Setup WebSocket server
const wss = new WebSocket.Server({ port: 8080 });


const childProcess = require('child_process');

let childProcesses = [];
let createdDirectories = [];

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
            // Read and parse the configuration file to find the persistence location
            const rl = readline.createInterface({
              input: fs.createReadStream(confFile),
              output: process.stdout,
              terminal: false
            });

            let persistenceLocation = '';
            rl.on('line', (line) => {
              if (line.startsWith('persistence_location')) {
                persistenceLocation = line.substring(line.indexOf(' ') + 1).trim();

                // Remove quotes from the persistence location if they exist
                if (persistenceLocation.startsWith('"') && persistenceLocation.endsWith('"')) {
                  persistenceLocation = persistenceLocation.slice(1, -1);
                }

                console.log(`folders to create ${persistenceLocation}`);
                // Create the directory if it does not exist
                if (!fs.existsSync(persistenceLocation)) {
                  fs.mkdirSync(persistenceLocation, { recursive: true });
                  createdDirectories.push(persistenceLocation); // add the created directory to the array
                }
              }
            });



            rl.on('close', () => {
              // Only spawn the mosquitto process after the configuration file has been read completely
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
                // Delete the persistence directory when the process exits
                if (fs.existsSync(persistenceLocation)) {
                  fs.rmdirSync(persistenceLocation, { recursive: true });
                }
              });
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

  // Delete .conf and .Identifier files in the current folder
  const currentFolder = process.cwd();
  const files = fs.readdirSync(currentFolder);
  files.forEach((file) => {
    const filePath = path.join(currentFolder, file);
    if (file.endsWith('.conf') || file.endsWith('.Identifier')) {
      fs.unlinkSync(filePath);
      console.log(`Deleted file: ${filePath}`);
    }
  });

  // Delete the created directories
  createdDirectories.forEach((dir) => {
    if (fs.existsSync(dir)) {
      fs.rmdirSync(dir, { recursive: true });
      console.log(`Deleted directory: ${dir}`);
    }
  });

  process.exit(); // Exit the server process
});
