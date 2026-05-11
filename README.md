# ComDeX NGSI-LD FastAPI Gateway

This project exposes ComDeX NGSI-LD operations through a FastAPI server.

ComDeX stores and exchanges NGSI-LD entity data over MQTT. The FastAPI layer makes the system easier to use from HTTP clients, Swagger UI, dashboards, scripts, and WebSocket clients.

## Table Of Contents

- [Project Overview](#project-overview)
- [Important Production Rule](#important-production-rule)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Setup](#setup)
- [Local Docker Broker Setup](#local-docker-broker-setup)
- [Production Broker Setup](#production-broker-setup)
- [Start The FastAPI Server](#start-the-fastapi-server)
- [Copy JSON Examples](#copy-json-examples)
- [Common Query Parameters](#common-query-parameters)
- [Create An Entity](#create-an-entity)
- [Query Entities](#query-entities)
- [Patch Full Entity Attributes](#patch-full-entity-attributes)
- [Patch One Attribute](#patch-one-attribute)
- [Delete One Attribute](#delete-one-attribute)
- [Delete An Entity](#delete-an-entity)
- [Batch Create](#batch-create)
- [Batch Update](#batch-update)
- [Batch Upsert](#batch-upsert)
- [Batch Delete](#batch-delete)
- [Create A Subscription By Type](#create-a-subscription-by-type)
- [Create A Subscription By Entity ID](#create-a-subscription-by-entity-id)
- [Create A Subscription With Watched Attributes](#create-a-subscription-with-watched-attributes)
- [List Subscriptions](#list-subscriptions)
- [Get One Subscription](#get-one-subscription)
- [Stop A Subscription](#stop-a-subscription)
- [WebSocket: Listen To Existing Subscription](#websocket-listen-to-existing-subscription)
- [WebSocket: Create Subscription And Stream In One Connection](#websocket-create-subscription-and-stream-in-one-connection)
- [Local Cross-Broker Test](#local-cross-broker-test)
- [Run The Test Suite](#run-the-test-suite)
- [Production Usage Checklist](#production-usage-checklist)
- [Troubleshooting](#troubleshooting)
- [Changes From The Original CLI Version](#changes-from-the-original-cli-version)

## Project Overview

ComDeX is an NGSI-LD data exchange layer built on top of MQTT. It stores entity attributes as MQTT retained messages and uses provider advertisement topics so other ComDeX nodes can discover where data is available.

The original ComDeX implementation is mainly a Python CLI tool. This version keeps the same MQTT-based ComDeX model, but exposes it through a FastAPI gateway so applications written in different languages can interact with ComDeX over standard HTTP and WebSocket interfaces.

The goal of this version is to make ComDeX easier to deploy, test, integrate, and operate in real client environments:

- HTTP clients can create, query, patch, delete, and batch-process NGSI-LD data without calling a Python CLI directly.
- WebSocket clients can receive live subscription notifications using a language-neutral interface.
- FastAPI Swagger UI can be used as an interactive testing console for all main operations.
- Docker-based Mosquitto brokers provide a reproducible local environment for testing single-broker and cross-broker flows.
- Runtime management was improved with reusable MQTT publisher clients, cleaner shutdown behavior, provider-child subscription control, and integration tests.

In short, this version turns the ComDeX CLI behavior into a service-oriented gateway suitable for API-based clients, demos, experiments, and production-style deployments.

## Important Production Rule

Do not use `localhost` when publishing real data that must be discovered by other machines.

When an entity is posted, ComDeX publishes a provider advertisement topic that includes the broker address:

```text
provider/<broker>/<port>/<area>/<context>/<entity-type>
```

If you POST with:

```text
broker = localhost
port = 1889
```

the advertisement becomes:

```text
provider/localhost/1889/unknown_area/https:.../GtfsAgency
```

Remote subscribers will then try to connect to their own `localhost`, not to your machine.

For production, POST/PATCH/batch data using the reachable machine IP or DNS name:

```text
broker = comdex-node-1.company.com
port = 1889
```

or:

```text
broker = 192.168.1.50
port = 1889
```

Use `localhost` only for local testing on one machine.

## Architecture

```text
Client / Swagger / Dashboard
        |
        | HTTP + WebSocket
        v
FastAPI server: actionhandlerAPI.py
        |
        | Python function calls
        v
ComDeX logic: actionhandler.py
        |
        | MQTT
        v
MQTT broker or broker network
```

Entity data is stored as MQTT retained messages. Subscriptions are held in FastAPI memory while the server is running.

If the FastAPI server restarts, active subscriptions must be recreated.

## Requirements

- Python 3.10 or newer
- Docker Desktop, if running the included Mosquitto brokers
- Network access from the FastAPI machine to the MQTT broker ports

## Setup

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Local Docker Broker Setup

The repository includes two Mosquitto brokers:

```text
broker1: localhost:1889
broker2: localhost:1890
```

`broker1` bridges only provider advertisement topics to `broker2`.

This is intentional. Entity data topics are not bridged. A remote subscriber first discovers a provider from the `provider/#` advertisement topic, then connects directly to the provider broker to receive the actual entity data.

Start them:

```bash
docker compose up -d
```

Check status:

```bash
docker compose ps
```

Stop them:

```bash
docker compose down
```

## Production Broker Setup

In production, the brokers may be deployed separately using Docker, Kubernetes, VMs, or managed MQTT infrastructure.

Make sure every broker advertised by ComDeX is reachable by other ComDeX nodes and subscribers.

Example production values:

```text
broker = comdex-node-1.company.com
port = 1889
qos = 1
```

If data is published to one broker and subscriptions are created on another broker, the brokers must be bridged or otherwise share provider advertisements.

Do not bridge all MQTT topics with `topic # both 0` for normal ComDeX operation. That would copy entity data between brokers and hide whether the subscription logic is really connecting to the advertised provider broker.

The local demo bridge is configured in:

```text
mosquitto/broker1/config/mosquitto.conf
```

Current local bridge:

```text
connection bridge-to-broker2
address broker2:1890
topic provider/# out 0
```

This means broker1 forwards provider advertisements to broker2. It does not forward:

```text
<area>/entities/...
<area>/Subscriptions/...
```

For the current ComDeX flow, only `provider/#` needs to be shared between brokers. Subscription metadata stays local to the API instance, and entity data remains on the provider broker.

For production, replace Docker service names and local ports with real reachable broker addresses.

Note: the current API accepts broker and port as query parameters. It does not yet expose MQTT username/password or TLS options. If the production broker requires authentication or TLS, the code must be extended.

## Start The FastAPI Server

Local testing:

```bash
uvicorn actionhandlerAPI:app --port 8000
```

Production-style bind:

```bash
uvicorn actionhandlerAPI:app --host 0.0.0.0 --port 8000
```

Do not use `--reload` when testing subscriptions or WebSockets. Reload restarts the API process and deletes in-memory subscriptions.

Open Swagger docs:

```text
http://127.0.0.1:8000/docs
```

If running on a server:

```text
http://<server-ip-or-dns>:8000/docs
```

## Copy JSON Examples

Every JSON request or response example in this README is written as a fenced `json` code block. On GitHub, each code block automatically shows a copy button in the top-right corner when you hover over it.

Click that copy button to copy the full JSON body directly to your clipboard, then paste it into FastAPI Swagger UI, Postman, curl, or your own client.

## Common Query Parameters

Most endpoints accept:

```text
broker = MQTT broker hostname or IP
port = MQTT broker port
qos = MQTT QoS level, usually 1
```

For local testing:

```text
broker = localhost
port = 1889
qos = 1
```

For production:

```text
broker = comdex-node-1.company.com
port = 1889
qos = 1
```

`hlink` is the context filter. Use:

```text
hlink = +
```

to match any context.

## Create An Entity

Endpoint:

```text
POST /ngsi-ld/v1/entities
```

Local query params:

```text
broker = localhost
port = 1889
qos = 1
```

Production query params:

```text
broker = <reachable-broker-dns-or-ip>
port = <broker-port>
qos = 1
```

Body:

```json
{
  "id": "urn:ngsi-ld:GtfsAgency:Malaga_EMT",
  "type": "GtfsAgency",
  "agencyName": {
    "type": "Property",
    "value": "Empresa Malaguena de Transportes"
  },
  "language": {
    "type": "Property",
    "value": "EN"
  },
  "page": {
    "type": "Property",
    "value": "http://www.emtmalaga.es/"
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

Expected response:

```json
{
  "status": "created",
  "id": "urn:ngsi-ld:GtfsAgency:Malaga_EMT"
}
```

If the entity already exists:

```json
{
  "detail": "Entity with this id already exists, did you mean to patch?"
}
```

## Query Entities

Endpoint:

```text
GET /ngsi-ld/v1/entities
```

Query by type:

```text
type = GtfsAgency
broker = localhost
port = 1889
hlink = +
limit = 1800
```

Query by id:

```text
id = urn:ngsi-ld:GtfsAgency:Malaga_EMT
broker = localhost
port = 1889
hlink = +
```

Return only selected attributes:

```text
type = GtfsAgency
attrs = agencyName,language
broker = localhost
port = 1889
hlink = +
```

Query by attribute value:

```text
type = GtfsAgency
q = language==EN
broker = localhost
port = 1889
hlink = +
```

## Patch Full Entity Attributes

Use this to update several attributes while keeping the same entity id.

Endpoint:

```text
PATCH /ngsi-ld/v1/entities/{entityId}/attrs
```

Params:

```text
entityId = urn:ngsi-ld:GtfsAgency:Malaga_EMT
broker = localhost
port = 1889
qos = 1
hlink = +
my_area = unknown_area
```

Body:

```json
{
  "agencyName": {
    "type": "Property",
    "value": "Empresa Malaguena de Transportes UPDATED"
  },
  "language": {
    "type": "Property",
    "value": "ES"
  },
  "timezone": {
    "type": "Property",
    "value": "Europe/Paris"
  }
}
```

Expected response:

```json
{
  "status": "updated",
  "id": "urn:ngsi-ld:GtfsAgency:Malaga_EMT"
}
```

Do not include `id`, `type`, or `@context` in this PATCH body.

## Patch One Attribute

Endpoint:

```text
PATCH /ngsi-ld/v1/entities/{entityId}/attrs/{attrName}
```

Params:

```text
entityId = urn:ngsi-ld:GtfsAgency:Malaga_EMT
attrName = language
broker = localhost
port = 1889
qos = 1
hlink = +
```

Body:

```json
{
  "type": "Property",
  "value": "FR"
}
```

Expected response:

```json
{
  "status": "updated",
  "id": "urn:ngsi-ld:GtfsAgency:Malaga_EMT",
  "attr": "language"
}
```

For this single-attribute endpoint, do not wrap the body inside `"language"`.

## Delete One Attribute

Endpoint:

```text
DELETE /ngsi-ld/v1/entities/{entityId}/attrs/{attrName}
```

Params:

```text
entityId = urn:ngsi-ld:GtfsAgency:Malaga_EMT
attrName = page
broker = localhost
port = 1889
hlink = +
```

Expected response:

```json
{
  "status": "deleted",
  "id": "urn:ngsi-ld:GtfsAgency:Malaga_EMT",
  "attr": "page"
}
```

## Delete An Entity

Endpoint:

```text
DELETE /ngsi-ld/v1/entities/{entityId}
```

Params:

```text
entityId = urn:ngsi-ld:GtfsAgency:Malaga_EMT
broker = localhost
port = 1889
hlink = +
my_area = unknown_area
```

Expected response:

```json
{
  "status": "deleted",
  "id": "urn:ngsi-ld:GtfsAgency:Malaga_EMT"
}
```

## Batch Create

Endpoint:

```text
POST /ngsi-ld/v1/entityOperations/create
```

Params:

```text
broker = localhost
port = 1889
qos = 1
my_area = unknown_area
my_loc = unknown_location
```

Body:

```json
[
  {
    "id": "urn:ngsi-ld:GtfsAgency:Batch_Agency_1",
    "type": "GtfsAgency",
    "agencyName": {
      "type": "Property",
      "value": "Batch Agency One"
    },
    "language": {
      "type": "Property",
      "value": "EN"
    },
    "@context": [
      "https://smartdatamodels.org/context.jsonld",
      "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
    ]
  },
  {
    "id": "urn:ngsi-ld:GtfsAgency:Batch_Agency_2",
    "type": "GtfsAgency",
    "agencyName": {
      "type": "Property",
      "value": "Batch Agency Two"
    },
    "language": {
      "type": "Property",
      "value": "FR"
    },
    "@context": [
      "https://smartdatamodels.org/context.jsonld",
      "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
    ]
  }
]
```

Expected response:

```json
{
  "status": "created",
  "count": 2
}
```

## Batch Update

Endpoint:

```text
POST /ngsi-ld/v1/entityOperations/update
```

Body:

```json
[
  {
    "id": "urn:ngsi-ld:GtfsAgency:Batch_Agency_1",
    "type": "GtfsAgency",
    "language": {
      "type": "Property",
      "value": "ES"
    },
    "@context": [
      "https://smartdatamodels.org/context.jsonld",
      "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
    ]
  },
  {
    "id": "urn:ngsi-ld:GtfsAgency:Batch_Agency_2",
    "type": "GtfsAgency",
    "language": {
      "type": "Property",
      "value": "IT"
    },
    "@context": [
      "https://smartdatamodels.org/context.jsonld",
      "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
    ]
  }
]
```

Expected response:

```json
{
  "status": "updated",
  "count": 2
}
```

## Batch Upsert

Endpoint:

```text
POST /ngsi-ld/v1/entityOperations/upsert
```

Upsert means:

```text
create if missing, update if already present
```

Use the same JSON array format as batch create/update.

Expected response:

```json
{
  "status": "upserted",
  "count": 2
}
```

## Batch Delete

Endpoint:

```text
POST /ngsi-ld/v1/entityOperations/delete
```

Params:

```text
broker = localhost
port = 1889
hlink = +
my_area = unknown_area
```

Body:

```json
[
  "urn:ngsi-ld:GtfsAgency:Batch_Agency_1",
  "urn:ngsi-ld:GtfsAgency:Batch_Agency_2"
]
```

Expected response:

```json
{
  "status": "deleted",
  "count": 2
}
```

## Create A Subscription By Type

Endpoint:

```text
POST /ngsi-ld/v1/subscriptions
```

For local cross-broker testing, subscribe on broker2:

```text
broker = localhost
port = 1890
qos = 1
my_area = unknown_area
```

Body:

```json
{
  "id": "urn:subscription:broker2-gtfs",
  "type": "Subscription",
  "entities": [
    {
      "type": "GtfsAgency"
    }
  ],
  "@context": [
    "https://smartdatamodels.org/context.jsonld",
    "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
  ]
}
```

Expected response:

```json
{
  "status": "created",
  "id": "urn:subscription:broker2-gtfs"
}
```

## Create A Subscription By Entity ID

Body:

```json
{
  "id": "urn:subscription:by-id-test",
  "type": "Subscription",
  "entities": [
    {
      "id": "urn:ngsi-ld:GtfsAgency:Malaga_EMT"
    }
  ],
  "@context": [
    "https://smartdatamodels.org/context.jsonld",
    "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
  ]
}
```

This receives notifications only for the matching entity id.

## Create A Subscription With Watched Attributes

Body:

```json
{
  "id": "urn:subscription:watched-attrs",
  "type": "Subscription",
  "entities": [
    {
      "type": "GtfsAgency"
    }
  ],
  "watchedAttributes": [
    "agencyName",
    "language"
  ],
  "@context": [
    "https://smartdatamodels.org/context.jsonld",
    "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
  ]
}
```

Only changes to `agencyName` and `language` will be sent as notifications.

## List Subscriptions

Endpoint:

```text
GET /ngsi-ld/v1/subscriptions
```

Example response:

```json
[
  {
    "id": "urn:subscription:broker2-gtfs",
    "type": "GtfsAgency",
    "entity_id": "",
    "broker": "localhost",
    "port": 1890,
    "created_at": "2026-05-04 16:25:39.161577",
    "expires": null,
    "status": "active"
  }
]
```

## Get One Subscription

Endpoint:

```text
GET /ngsi-ld/v1/subscriptions/{subscriptionId}
```

Example:

```text
subscriptionId = urn:subscription:broker2-gtfs
```

## Stop A Subscription

Endpoint:

```text
DELETE /ngsi-ld/v1/subscriptions/{subscriptionId}
```

Example:

```text
subscriptionId = urn:subscription:broker2-gtfs
```

Expected response:

```json
{
  "status": "stopped",
  "id": "urn:subscription:broker2-gtfs"
}
```

Connected WebSocket clients receive:

```json
{
  "status": "stopped",
  "id": "urn:subscription:broker2-gtfs"
}
```

and the WebSocket closes cleanly.

## WebSocket: Listen To Existing Subscription

Swagger UI does not provide a good WebSocket listener. Use a small script.

Create `websocketlistener.py`:

```python
import asyncio
import json
import websockets
from websockets.exceptions import ConnectionClosed

SUBSCRIPTION_ID = "urn:subscription:broker2-gtfs"

async def main():
    uri = f"ws://127.0.0.1:8000/ngsi-ld/v1/subscriptions/{SUBSCRIPTION_ID}/ws"

    try:
        async with websockets.connect(uri) as ws:
            while True:
                msg = await ws.recv()
                print("Notification received:")
                print(msg)

                data = json.loads(msg)
                if data.get("status") == "stopped":
                    break

    except ConnectionClosed as e:
        print(f"WebSocket closed: code={e.code}, reason={e.reason}")

asyncio.run(main())
```

Run:

```bash
python websocketlistener.py
```

Expected first message:

```json
{
  "status": "connected",
  "id": "urn:subscription:broker2-gtfs"
}
```

The server also sends an initial snapshot of already-existing matching entities, then continues sending live notifications.

## WebSocket: Create Subscription And Stream In One Connection

Endpoint:

```text
WS /ngsi-ld/v1/subscriptions/ws
```

Flow:

```text
1. Connect to WebSocket.
2. Send subscription JSON.
3. Receive {"status":"subscribed","id":"..."}.
4. Receive initial/live entity notifications.
5. Disconnect to automatically stop the subscription.
```

## Local Cross-Broker Test

1. Start brokers:

```bash
docker compose up -d
```

2. Start API:

```bash
uvicorn actionhandlerAPI:app --port 8000
```

3. POST entity to broker1:

```text
POST /ngsi-ld/v1/entities
broker = localhost
port = 1889
qos = 1
```

4. Create subscription on broker2:

```text
POST /ngsi-ld/v1/subscriptions
broker = localhost
port = 1890
qos = 1
```

5. Start WebSocket listener:

```bash
python websocketlistener.py
```

6. PATCH entity on broker1:

```text
PATCH /ngsi-ld/v1/entities/{entityId}/attrs
broker = localhost
port = 1889
qos = 1
```

7. The WebSocket listener should receive a notification.

## Run The Test Suite

The integration unittest suite checks the main ComDeX FastAPI flows against the Docker Mosquitto brokers:

- Entity POST, GET, PATCH, attribute DELETE, and entity DELETE
- PATCH full entity and single attribute with `validate_exists=true` and `validate_exists=false`
- Batch upsert, update, and delete
- Subscription WebSocket delivery after POST and PATCH
- Subscriptions by type, by entity ID, with watched attributes, and without watched attributes
- List/get subscription endpoints
- Provider child subscription removal
- Parent subscription stop and clean WebSocket close

Start the brokers first:

```bash
docker compose up -d
```

Then run:

```bash
python -m unittest tests.test_comdex_api -v
```

If FastAPI is not already running on `127.0.0.1:8000`, the tests start it temporarily and stop it when finished.

## Production Usage Checklist

Before using this in production:

- Use real broker IP/DNS when posting data.
- Ensure advertised broker hostnames are reachable by remote ComDeX nodes.
- Open firewall ports for MQTT and API traffic.
- Do not use `uvicorn --reload`.
- Decide how subscriptions are recreated after API restart.
- Add MQTT authentication/TLS support if the broker requires it.
- Put FastAPI behind a reverse proxy if exposing it to users.
- Restrict access to `/docs` in public deployments.

## Troubleshooting

Duplicate POST returns:

```json
{
  "detail": "Entity with this id already exists, did you mean to patch?"
}
```

Use PATCH to update existing entities.

If WebSocket returns `403`, the subscription probably does not exist in memory. Recreate the subscription and reconnect.

If a remote subscriber cannot receive data, check the provider advertisement. It must contain a reachable broker address, not `localhost`.

If `GET /entities` returns empty, confirm:

- The broker and port are correct.
- The entity was posted to that broker.
- `hlink = +` is used for simple tests.
- The broker retained messages were not cleared.

If cross-broker subscription does not work, confirm:

- Broker bridge/federation is configured.
- Provider topics are visible on the subscription broker.
- QoS is set to `1`.

## Changes From The Original CLI Version

This version is based on the original ComDeX CLI action handler from:

```text
https://github.com/satrai-lab/comdex/blob/main/actionhandler.py
```

The original project exposes ComDeX mainly as a Python command-line tool. This version keeps the MQTT/NGSI-LD topic model, retained messages, provider advertisements, entity operations, and subscription logic, then adds an HTTP/WebSocket gateway and production-oriented runtime behavior.

Main modifications added in this version:

- Added `actionhandlerAPI.py`, a FastAPI layer exposing NGSI-LD-style HTTP endpoints.
- Added Swagger/OpenAPI testing through FastAPI docs at `/docs`.
- Added HTTP endpoints for:
  - `POST /ngsi-ld/v1/entities`
  - `GET /ngsi-ld/v1/entities`
  - `PATCH /ngsi-ld/v1/entities/{entityId}/attrs`
  - `PATCH /ngsi-ld/v1/entities/{entityId}/attrs/{attrName}`
  - `DELETE /ngsi-ld/v1/entities/{entityId}`
  - `DELETE /ngsi-ld/v1/entities/{entityId}/attrs/{attrName}`
  - batch create, update, upsert, and delete operations
  - subscription create, list, get, stop, and provider-child stop operations
- Added WebSocket notification endpoints so clients in any language can subscribe and receive live data:
  - `WS /ngsi-ld/v1/subscriptions/{subscriptionId}/ws`
  - `WS /ngsi-ld/v1/subscriptions/ws`
- Added `websocketlistener.py` as a simple client example for consuming subscription notifications.
- Added Docker-based Mosquitto broker setup with `docker-compose.yml` and broker config folders.
- Added support for two local brokers, useful for testing provider advertisements and cross-broker subscriptions.
- Changed entity POST behavior to upsert-style publishing: posting an entity with an existing ID updates the retained entity data instead of failing as a duplicate.
- Kept advertisement protection: provider advertisements are checked before publishing, so an existing advertisement is not republished unnecessarily.
- Added reusable MQTT publisher clients for POST/PATCH paths to reduce repeated MQTT connection overhead.
- Added optional PATCH existence validation:
  - `validate_exists=true` keeps the safer behavior and checks that the entity exists before patching.
  - `validate_exists=false` skips the existence check for faster updates when the client already knows the entity exists, but requires `hlink` and `entity_type`.
- Added provider child process tracking for subscriptions.
- Added an endpoint to stop only one provider child subscription while keeping the parent subscription active:
  - `DELETE /ngsi-ld/v1/subscriptions/{subscriptionId}/providers`
- Added protection so a stopped provider child is not immediately recreated from a retained provider advertisement.
- Added queue filtering so WebSocket clients do not receive new notifications from a provider after that provider child subscription has been disabled.
- Added clean parent subscription stop behavior: WebSocket clients receive a final `{"status": "stopped", "id": ...}` message before the connection closes.
- Added FastAPI shutdown cleanup so Ctrl+C stops active subscription threads/processes and closes MQTT publisher clients.
- Added a production-focused README with setup, API usage, WebSocket usage, Docker broker usage, and troubleshooting.
- Added `tests/test_comdex_api.py`, an integration unittest suite covering entity lifecycle, PATCH variants, batch operations, subscriptions, WebSockets, provider child deletion, and clean shutdown behavior.
