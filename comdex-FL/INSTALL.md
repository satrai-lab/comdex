# INSTALL

The prototype assumes the presence of at least one running MQTT broker. For federation experiments, multiple brokers with bridge configurations can be used.

------------------------------------------------------------------------

## Option A (Recommended): Local Python Environment + Local Broker

### Prerequisites

-   Python 3.10+ (3.11 tested)
-   Mosquitto MQTT broker
-   Linux/macOS (Windows works, but examples assume bash)

Install Mosquitto:

**Ubuntu/Debian**

``` bash
sudo apt update
sudo apt install mosquitto mosquitto-clients
```

**macOS (Homebrew)**

``` bash
brew install mosquitto
```

------------------------------------------------------------------------

## Step 1 --- Start a Local Broker

Start a simple broker instance:

In order to use the sanity check script prefer using the provided configuration file:

``` bash
mosquitto -c ./configs/mosquitto/mosquitto_generic.conf -v 
```

Leave this terminal running.

------------------------------------------------------------------------

## Step 2 --- Create Python Environment

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Step 3 --- Run Sanity Check

With the broker running:

``` bash
python3 sanity_check.py
```

If it looks stuck, don't worry, it just takes a little time to run, you should be s

Expected behaviour:

-   The script connects to the local broker
-   Publishes a test coordination entity
-   Subscribes to coordination topics
-   Confirms message exchange
-   Training happens and you can see server and client output

If successful, you should see connection logs and confirmation messages
printed in the terminal.

------------------------------------------------------------------------

## Optional: Quick Federation Setup (Multiple Brokers)

To emulate multiple domains:

1.  Start Broker A:

    ``` bash
    mosquitto -c ./configs/mosquitto/broker_a.conf -v
    ```

2.  Start Broker B:

    ``` bash
    mosquitto -c ./configs/mosquitto/broker_b.conf -v
    ```

3.  Ensure bridge settings are configured in the `.conf` files (see
    provided examples).

This allows cross-broker topic propagation and enables multi-domain FL
coordination experiments.

------------------------------------------------------------------------

## Notes

-   Ensure no firewall blocks local connections.
-   If any of the ports used in the configs are already used in your machine, change the port in the config and
    update the Python client accordingly.

------------------------------------------------------------------------