# ComDeX Federated Learning (FL) Extension

Here you can find the federated learning extension of [ComDeX].

This extension enables smart communities to collaboratively train machine learning models across distributed datasets **without exposing raw data**. It builds upon ComDeX’s existing advertisement-based architecture to support:

- **Dynamic FL Task Discovery**: FL tasks and FL Participants can be advertised and discovered across brokers.
- **Flexible Participant Enrollment**: Supports client-driven, server-driven, and hybrid discovery paradigms.
- **Federated Training**: Integrates with the [Flower](https://flower.ai/) framework to perform secure model training and aggregation.

It is mainly a **proof-of-concept prototype** focused on **cross-organisation Federated Learning (FL)**.
It demonstrates (i) task publication and discovery via NGSI-LD–style entities, (ii) coordination via a federated
broker layer (MQTT with bridging), and (iii) FL execution using a standard FL runtime. In the prototype,
**feasibility and negotiation are operator-driven**: an experimenter inspects the discovered candidates and
starts execution for the chosen collaboration plan. (LINK TO CODE: https://zenodo.org/records/18631547) 


### Note: Make sure you are using the zenodo version linked in the artifact page to download the repository.

## What is included

- `fl_collab.py`: orchestration entry point for the FL proof-of-concept (quickstart + paper-mode template).
- `c_action.py`: helper logic for publishing and consuming task/offer metadata over the broker layer (basically comdex action-handler).
- `sanity_check.py`: minimal self-test (validates local environment).
- `configs/mosquitto/`: example Mosquitto configurations for a small federated broker topology via MQTT bridges.

## Repository structure

```
.
├── configs/mosquitto/         # broker configs (MQTT bridges)
├── data/                     # local datasets / cached downloads
├── models                    # local trained model storage
├── results/                  # output produced by past runs
├── INSTALL.md                # step-by-step installation
└── README.md                 # this file
```

## System overview (conceptual)

The prototype follows the architecture’s main loop:

1. **Publish**: the initiator publishes a task intent (FL training job) as a structured entity.
2. **Discover**: participants publish offers; discovery queries match intents to offers.
3. **Feasibility (manual in prototype)**: the operator reviews discovered candidates and selects participants.
4. **Execute**: FL training runs across selected participants; results and metrics are recorded.

The broker layer is intentionally simple: it provides federation and discovery through publish/subscribe
dissemination and broker-to-broker bridges.

## Getting started 

1. Follow `INSTALL.md` to create a Python environment and install dependencies.
2. You can see a simple run using the final script present in the intall.md :
   ```bash
   python src/sanity_check.py
   ```
3. Feel free to modify it as you see fit if you want to test functionality on a single broker.   

# Local Federated Setup and Experiment Reproduction Guide

This guide explains how to:

1. Spin up the three MQTT brokers (A → B → C)
2. Create a subscription for FL job discovery
3. Start the FL server (publishes job on Broker A)
4. Discover the server from Broker C
5. Start a client from Broker C

This setup enables a local federation on a single machine.

---

# 1. Broker Topology

We use three Mosquitto brokers bridged in a chain:

A → B → C

Ports:

- Broker A: localhost:1026  
- Broker B: localhost:1028  
- Broker C: localhost:1029  

---

# 2. Starting the Brokers

From the root of the repository:

## Broker A

mosquitto -c ./configs/mosquitto/mosquitto-a.conf -v

## Broker B

mosquitto -c ./configs/mosquitto/mosquitto-b.conf -v

## Broker C

mosquitto -c ./configs/mosquitto/mosquitto-c.conf -v

Run each in a separate terminal.

The configuration files define the bridging such that messages published on A propagate to B and then to C.

---

### 3. Create a Subscription on Broker C (Discovery Step)

The client discovers FL jobs by subscribing to FederatedLearningJob entities.

Subscription file example (subscription_example.ngsild):

{
  "id": "urn:subscription:5",
  "type": "Subscription",
  "entities": [
    {
      "type": "FederatedLearningJob"
    }
  ],
  "@context": [
    "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
  ]
}

Create the subscription on Broker C (port 1029):

sudo python3 c_action.py   -c POST/Subscriptions   -f example_subscription.ngsild   -b localhost   -p 1029

After this step, Broker C will output notifications whenever a
FederatedLearningJob entity is published or updated upstream.

---

### 4. Start the FL Server (Publishes Job to Broker A)

Example: CIFAR-10, 1 round, FedAvg

python3 fl_collab.py   --role server   --membership-approach client_driven   --broker-url mqtt://localhost:1026   --server-address localhost:1027   --strategy-name fedavg   --job-id test_cifar   --num-rounds 1   --dataset cifar10   --model-type tinycnn   --required-participants 1

What happens:

- The server publishes a FederatedLearningJob entity to Broker A (1026).
- The entity propagates A → B → C.
- Broker C prints entity fragments including:
  - serverAddress
  - numRounds
  - strategy
  - modelSpec
  - requiredParticipants

The important value is:

serverAddress.value → localhost:1027

This is the Flower server endpoint.

---

### 5. Start a Client from Broker C

Now start a client that connects to the discovered server address.

python3 fl_collab.py   --role client   --membership-approach client_driven   --broker-url mqtt://localhost:1029   --server-address localhost:1027   --client-id client_0   --dataset cifar10   --model-type tinycnn   --num-clients 1

Expected behavior:

- Client advertises its capability to Broker C
- Client connects to Flower server at localhost:1027
- Training round executes
- Server publishes GlobalModel context updates back to brokers

---

### 6. Reproducing Full Paper Experiments

The paper experiments used multiple machines (e.g., cloud instances) hosting:

- Separate brokers per federation
- Separate FL servers
- Separate client groups

The local federation setup described here demonstrates:

- Cross-broker job advertisement
- Discovery via NGSI-LD entities
- Cross-domain Flower coordination
- Context updates of global model rounds

For large-scale experiments:

- Deploy brokers on separate machines
- Run server on Machine A
- Run clients on Machines B and C
- Keep bridging topology consistent

---

###  Notes

- Ensure ports 1026, 1028, 1029, 1027, and 8000 are free.
- Run brokers before starting server or clients.
- In client-driven mode, the client learns the server address from the broker layer.
- The system does not require centralized coordination outside the broker federation.

# Reproducing the paper figures

This artifact supports two levels of reproduction:

1. **Functional reproduction on a single machine** (local broker federation + one server + one or more clients), to validate end-to-end task advertisement, discovery, and FL execution.
2. **Paper-style reproduction** (multi-machine / multi-VM deployment) to mirror the paper’s distributed setup (separate brokers/clients/server across hosts).

### Where runs write results

Each FL server run logs **per-round metrics** (e.g., round number, timestamp, loss, accuracy, round duration, number of clients, failures) as a CSV file named:

- `models/<JOB_ID>_round_metrics.csv`

The `models/` directory also contains saved global checkpoints (when enabled), e.g.:

- `models/global_round_<ROUND>.pt`

### Using the included CSVs and regenerating plots

The repository includes **CSV result files from prior runs** under `results/` (for convenience and to enable fast plotting without rerunning training).

To reproduce the figures from the paper using these CSVs:

```bash
cd results
python3 analysis.py
```

The script reads the CSV files under `results/` (and/or `../models/` depending on configuration) and regenerates the plots used in the paper. The produced figures are written to the `results/` directory (see script output for exact filenames).

Example Output:

![Highest Accuracy Per Strategy](./results/Highest_Accuracy_Per_Strategy.png)
