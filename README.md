# ComDeX Prototype Implementation


## Overview
ComDeX is a lightweight, federated NGSI-LD broker system that makes use of open-source MQTT brokers at its core. It was concepted originally as a simple test suite for using MQTT with [NGSI-LD](https://www.etsi.org/deliver/etsi_gs/CIM/001_099/009/01.01.01_60/gs_CIM009v010101p.pdf) and evolved over time to be a capable Internet of Things (IoT) platform.
Our prototype provides an alternative solution to existing heavyweight NGSI-LD brokers, which generally favour HTTP as a communication protocol. ComDeX provides end-to-end MQTT capabilities, such as QoS delivery guarantees, that aren't possible with these solutions.

For more detailed information about the inner workings of our prototype as well as working examples, visit our wiki [link here TBA].

## How It Works

The ComDeX platform is a federation of ComDeX nodes.

Each ComDeX node has two main key components: The Action Handler and an MQTT Broker.

### Action Handler
The Action Handler is an API for various clients (producers/consumers) to conduct diverse "Actions". These "Actions" are defined as any operation inside the architecture that is necessary for the exchange of information between clients and the brokers.
This component provides high-level functions for data context discovery, both synchronous and asynchronous. It is responsible for executing commands and managing data flows using various sub-components. Simply put, it's the thing that turns ComDeX into ComDeX. While the ComDeX architecture is not targeted torwards only NGSI-LD, for the implementation of the prototype, since NGSI-LD has been used, there has been an effort for the endpoints to be as much as possible NGSI-LD like

### MQTT Broker
This is the core of each ComDeX node. Although we've extensively used Mosquitto for our testing, our solution is not limited to this particular MQTT broker. You can use any MQTT broker you prefer, as long as it offers the persistence of messages and creation of MQTT bridges, which are essential for federation.

## Installation
### Requirements
Our ComDeX prototype implementation is in Python but can be easily adapted to other programming languages.

Here are the requirements you need for the installation:

- Python environment.
- An MQTT broker with capabilities for message persistence and creating MQTT bridges.


For the Action Handler, you'll need the following libraries:

```
paho-mqtt==1.6.1
Shapely==1.8.1
```
These are included in the requirements.txt file.

How to Install Required Libraries
To install the required libraries, you will need to run the following command in your terminal:

```
pip install -r requirements.txt
```

This command will install all the libraries listed in the requirements.txt file.

Running the Action Handler
To see the list of available command-line arguments and their use, run the action_handler.py with the -h flag.

```
python action_handler.py -h
```


## Limitations
As a new system and an ongoing project, ComDeX is not a mature solution that has been thoroughly tested. Its current state reflects a prototype implementation of our original idea. The exact compliance with the NGSI-LD specification hasn't been tested.

## Contribution
Contributions are welcome. Please feel free to contribute by submitting a pull request.


## Contact
For any further questions, please feel free to reach us at [papadakni@ics.forth.gr].

## Acknowledgements
This work is partly based on the [NGSI-LD specification](https://www.etsi.org/deliver/etsi_gs/CIM/001_099/009/01.01.01_60/gs_CIM009v010101p.pdf) developed by ETSI.
We thankfully acknowledge funding for this research by the Greek RTDI Action “RESEARCH-CREATE-INNOVATE” (EΠA𝜈EK 2014-2020), Grant no. T2EΔK-02848 (SmartCityBus).

Please note that the information in this README is subject to changes as the project evolves.
