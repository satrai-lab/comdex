# ComDeX Prototype Implementation

<div style="display: flex;">
  <img src="https://www.cidoc-crm.org/sites/default/files/ics-diskout-en.jpg" alt="logo1" width="500" height="150" style="border-right: 2px solid #ccc; padding-right: 10px;">
  <img src="https://s3-eu-west-1.amazonaws.com/assets.atout-on-line.com/images/ingenieur/Logos_Ecoles/2018_2021/telecom_sudparis_300.jpg" alt="logo2" width="200" height="300">
</div>

This work is the result of combined research from [ICS-FORTH](https://www.ics.forth.gr/) and [Telecom SudParis](https://www.telecom-sudparis.eu/).

An agile and decentralized system designed for the dynamic evolution of the Internet of Things (IoT)

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Limitations](#limitations)
- [Contribution](#contribution)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Overview

ComDeX is a lightweight, federated NGSI-LD broker system that leverages open-source MQTT brokers. Initially conceived as a test suite for MQTT with [NGSI-LD](https://www.etsi.org/deliver/etsi_gs/CIM/001_099/009/01.01.01_60/gs_CIM009v010101p.pdf), it has evolved into a versatile Internet of Things (IoT) platform. Compared to traditional heavyweight NGSI-LD brokers favoring HTTP as a communication protocol, ComDeX offers end-to-end MQTT capabilities, including QoS delivery guarantees. For a deep dive into our prototype and working examples, refer to our wiki (link TBA).

## How It Works

The ComDeX platform operates as a federation of ComDeX nodes, each consisting of two primary components: the Action Handler and an MQTT Broker.

### Action Handler

The Action Handler is an API that enables various clients (producers/consumers) to perform diverse "Actions," defined as any operation within the architecture essential for information exchange between clients and brokers. This component facilitates data context discovery (both synchronous and asynchronous) and manages data flows. While ComDeX isn't strictly designed for NGSI-LD, our prototype implementation tries to adhere to NGSI-LD endpoints as closely as possible.

### MQTT Broker

This component serves as the backbone of each ComDeX node. While Mosquitto is extensively used for testing, our solution isn't confined to this MQTT broker. You can use any MQTT broker, provided it supports message persistence and MQTT bridges creation—two critical features for federation.

## Installation

### Requirements

Our ComDeX prototype implementation is written in Python, though it's easily adaptable to other programming languages.

- Python environment
- MQTT broker supporting message persistence and MQTT bridges creation

For the Action Handler, you'll need the following libraries:

```
paho-mqtt==1.6.1
Shapely==1.8.1
```
These requirements are included in the requirements.txt file.

### Installation Steps
Install the required libraries with the command: pip install -r requirements.txt.
To view the list of available command-line arguments and their usage, execute 

```python
python3 action_handler.py -h.
```

### Sanity Check
To do a quick sanity check that everything has been setup correctly you can do the following:
In the same folder as "actionhandler.py" create 2 files:
  An entity example file, "entity_example.ngsild":
  ```
  {
    "id": "urn:ngsi-ld:GtfsAgency:Malaga_EMT",
    "type": "GtfsAgency",
    "agencyName": {
        "type": "Property",
        "value": "Empresa Malague\u00f1a de Transportes"
    },
    "language": {
        "type": "Property",
        "value": "ES"
    },
    "page": {
        "type": "Property",
        "value": "http://www.emtmalaga.es/"
    },
    "source": {
        "type": "Property",
        "value": "http://datosabiertos.malaga.eu/dataset/lineas-y-horarios-bus-google-transit/resource/24e86888-b91e-45bf-a48c-09855832fd52"
    },
    "timezone": {
        "type": "Property",
        "value": "Europe/Madrid"
    },
    "@context": [
    "https://smartdatamodels.org/context.jsonld",
    "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
    ]
}
  ```
  And a subscription file, subscription_example.ngsild:
  ```
  {
  "id": "urn:subscription:3",
  "type": "Subscription",
  "entities": [{
                "type": "GtfsAgency"
  }],
  "watchedAttributes": ["agencyName","language"],
  "@context": [
    "https://smartdatamodels.org/context.jsonld",
    "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
    ]
}
  ```

We are going to create an entity for which we have subscribed to some of its attributes.
We can do this in any order we like (either publish before or after the subscription).
Replace localhost and 1026 with the appropriate address and port of your MQTT broker.

```
sudo python3 actionhandler.py -c POST/Subscriptions -f subscription_example.ngsild -b localhost -p 1026
sudo python3 actionhandler.py -c POST/entities -f entity_example.ngsild -b localhost -p 1026
```
You should be able to see the subscribed attributes printed/returned at your terminal

You can also try "getting" the published entity, using the following command:
```
sudo python3 actionhandler.py -c GET/entities/?id=urn:ngsi-ld:GtfsAgency:Malaga_EMT -b localhost -p 1026
```

## Limitations
As a nascent system and an ongoing project, ComDeX isn't a mature solution yet and hasn't been thoroughly tested. It currently serves as a prototype implementation of our original idea, and the exact compliance with the NGSI-LD specification remains untested.

## Contribution
We heartily welcome contributions! Feel free to submit a pull request.

## Contact
For further inquiries, feel free to reach us at [papadakni@ics.forth.gr].

## Acknowledgements
This work is partly based on the NGSI-LD specification developed by ETSI. Our sincere thanks to the Greek RTDI Action “RESEARCH-CREATE-INNOVATE” (EΠA𝜈EK 2014-2020), Grant no. T2EΔK-02848 (SmartCityBus), for funding this research.

Note: This README is subject to changes as the project progresses.
