import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import unittest
import uuid
from pathlib import Path
from urllib.parse import quote

import paho.mqtt.client as mqtt
import requests
import websockets


BASE_URL = os.getenv("COMDEX_BASE_URL", "http://127.0.0.1:8000")
WS_BASE_URL = BASE_URL.replace("http://", "ws://").replace("https://", "wss://")
BROKER_HOST = os.getenv("COMDEX_TEST_BROKER", "localhost")
BROKER1_PORT = int(os.getenv("COMDEX_TEST_BROKER1_PORT", "1889"))
BROKER2_PORT = int(os.getenv("COMDEX_TEST_BROKER2_PORT", "1890"))
AREA = os.getenv("COMDEX_TEST_AREA", "unknown_area")
QOS = int(os.getenv("COMDEX_TEST_QOS", "0"))
CONTEXT = [
    "https://smartdatamodels.org/context.jsonld",
    "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld",
]
PRIMARY_CONTEXT = CONTEXT[0]


def wait_for_tcp(host, port, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def api_is_ready(base_url=BASE_URL):
    try:
        response = requests.get(f"{base_url}/openapi.json", timeout=1)
        return response.status_code == 200
    except requests.RequestException:
        return False


def wait_for_api(timeout=15.0, base_url=BASE_URL):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if api_is_ready(base_url):
            return True
        time.sleep(0.25)
    return False


class ComdexApiIntegrationTests(unittest.TestCase):
    server_process = None

    @classmethod
    def setUpClass(cls):
        if not wait_for_tcp(BROKER_HOST, BROKER1_PORT) or not wait_for_tcp(BROKER_HOST, BROKER2_PORT):
            raise unittest.SkipTest(
                "Mosquitto brokers are not reachable. Start them with: docker compose up -d"
            )

        if not api_is_ready():
            root = Path(__file__).resolve().parents[1]
            cls.server_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "actionhandlerAPI:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8000",
                ],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if not wait_for_api():
                cls._stop_server()
                raise RuntimeError("FastAPI server did not start on port 8000")

        cls.session = requests.Session()

    @classmethod
    def tearDownClass(cls):
        cls.session.close()
        cls._stop_server()

    @classmethod
    def _stop_server(cls):
        if cls.server_process is None:
            return
        cls.server_process.terminate()
        try:
            cls.server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.server_process.kill()
            cls.server_process.wait(timeout=5)
        cls.server_process = None

    def unique(self, label):
        return f"{label}{uuid.uuid4().hex[:8]}"

    def get_entities_with_retry(self, path, retries=3, delay=1.0):
        """GET entities, retrying up to `retries` times if the result is empty.

        The GET function in actionhandler.py waits at most 1 second for MQTT retained
        messages.  Under broker load that window can be missed, producing a spurious
        empty result.  Retrying absorbs these transient misses without hiding real bugs
        (a genuine empty result fails after all retries are exhausted).
        """
        result = []
        for _ in range(retries):
            result = self.request_json("GET", path)
            if result:
                break
            time.sleep(delay)
        return result

    def request_json(self, method, path, expected_status=200, **kwargs):
        response = self.session.request(method, f"{BASE_URL}{path}", timeout=15, **kwargs)
        self.assertEqual(
            expected_status,
            response.status_code,
            msg=f"{method} {path} returned {response.status_code}: {response.text}",
        )
        if not response.content:
            return None
        return response.json()

    def entity_payload(self, entity_type, entity_id, name="ComDeX Test Agency", language="EN"):
        return {
            "id": entity_id,
            "type": entity_type,
            "agencyName": {"type": "Property", "value": name},
            "language": {"type": "Property", "value": language},
            "@context": CONTEXT,
        }

    def post_entity(self, entity, port=BROKER1_PORT):
        return self.request_json(
            "POST",
            f"/ngsi-ld/v1/entities?broker={BROKER_HOST}&port={port}&qos={QOS}&my_area={AREA}",
            expected_status=201,
            json=entity,
        )

    def delete_entity_best_effort(self, entity_id, port=BROKER1_PORT):
        try:
            self.session.delete(
                f"{BASE_URL}/ngsi-ld/v1/entities/{quote(entity_id, safe='')}"
                f"?broker={BROKER_HOST}&port={port}&hlink={PRIMARY_CONTEXT}&my_area={AREA}",
                timeout=10,
            )
        except requests.RequestException:
            pass

    def delete_subscription_best_effort(self, subscription_id):
        try:
            self.session.delete(
                f"{BASE_URL}/ngsi-ld/v1/subscriptions/{quote(subscription_id, safe='')}",
                timeout=10,
            )
        except requests.RequestException:
            pass

    async def recv_json(self, websocket, timeout=5.0):
        raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
        return raw, json.loads(raw)

    async def wait_for_entity_message(self, websocket, entity_id, timeout=8.0, predicate=None):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            raw, message = await self.recv_json(websocket, remaining)
            if isinstance(message, dict) and message.get("id") == entity_id:
                if predicate is None or predicate(message):
                    return raw, message
        raise AssertionError(f"No matching WebSocket entity message received for {entity_id}")

    def test_01_entity_lifecycle_post_get_patch_delete(self):
        entity_type = self.unique("ComdexLifecycle")
        entity_id = f"urn:ngsi-ld:{entity_type}:001"
        self.addCleanup(self.delete_entity_best_effort, entity_id, BROKER1_PORT)

        created = self.post_entity(self.entity_payload(entity_type, entity_id), BROKER1_PORT)
        self.assertEqual("upserted", created["status"])
        self.assertEqual(entity_id, created["id"])

        entities = self.get_entities_with_retry(
            f"/ngsi-ld/v1/entities?type={entity_type}&broker={BROKER_HOST}"
            f"&port={BROKER1_PORT}&hlink={PRIMARY_CONTEXT}",
        )
        self.assertTrue(any(entity.get("id") == entity_id for entity in entities))

        patched = self.request_json(
            "PATCH",
            f"/ngsi-ld/v1/entities/{quote(entity_id, safe='')}/attrs"
            f"?broker={BROKER_HOST}&port={BROKER1_PORT}&qos={QOS}"
            f"&hlink={PRIMARY_CONTEXT}&my_area={AREA}",
            json={"language": {"type": "Property", "value": "FR"}},
        )
        self.assertEqual("updated", patched["status"])

        entities = self.get_entities_with_retry(
            f"/ngsi-ld/v1/entities?type={entity_type}&broker={BROKER_HOST}"
            f"&port={BROKER1_PORT}&hlink={PRIMARY_CONTEXT}",
        )
        matching_entities = [entity for entity in entities if entity.get("id") == entity_id]
        self.assertEqual(1, len(matching_entities))
        self.assertEqual("FR", matching_entities[0]["language"]["value"])

        deleted_attr = self.request_json(
            "DELETE",
            f"/ngsi-ld/v1/entities/{quote(entity_id, safe='')}/attrs/language"
            f"?broker={BROKER_HOST}&port={BROKER1_PORT}&hlink={PRIMARY_CONTEXT}&my_area={AREA}",
        )
        self.assertEqual("deleted", deleted_attr["status"])

        deleted = self.request_json(
            "DELETE",
            f"/ngsi-ld/v1/entities/{quote(entity_id, safe='')}"
            f"?broker={BROKER_HOST}&port={BROKER1_PORT}&hlink={PRIMARY_CONTEXT}&my_area={AREA}",
        )
        self.assertEqual("deleted", deleted["status"])

    def test_02_batch_upsert_update_delete(self):
        entity_type = self.unique("ComdexBatch")
        ids = [f"urn:ngsi-ld:{entity_type}:{index}" for index in range(2)]
        for entity_id in ids:
            self.addCleanup(self.delete_entity_best_effort, entity_id, BROKER1_PORT)

        entities = [
            self.entity_payload(entity_type, ids[0], name="Batch Agency 1", language="EN"),
            self.entity_payload(entity_type, ids[1], name="Batch Agency 2", language="EN"),
        ]
        upserted = self.request_json(
            "POST",
            f"/ngsi-ld/v1/entityOperations/upsert?broker={BROKER_HOST}&port={BROKER1_PORT}"
            f"&qos={QOS}&my_area={AREA}",
            json=entities,
        )
        self.assertEqual(2, upserted["count"])

        updated_entities = [
            self.entity_payload(entity_type, ids[0], name="Batch Agency 1 Updated", language="ES"),
            self.entity_payload(entity_type, ids[1], name="Batch Agency 2 Updated", language="ES"),
        ]
        updated = self.request_json(
            "POST",
            f"/ngsi-ld/v1/entityOperations/update?broker={BROKER_HOST}&port={BROKER1_PORT}"
            f"&qos={QOS}&my_area={AREA}",
            json=updated_entities,
        )
        self.assertEqual(2, updated["count"])

        fetched = self.request_json(
            "GET",
            f"/ngsi-ld/v1/entities?type={entity_type}&broker={BROKER_HOST}"
            f"&port={BROKER1_PORT}&hlink={PRIMARY_CONTEXT}",
        )
        fetched_ids = {entity["id"] for entity in fetched}
        self.assertTrue(set(ids).issubset(fetched_ids))

        deleted = self.request_json(
            "POST",
            f"/ngsi-ld/v1/entityOperations/delete?broker={BROKER_HOST}&port={BROKER1_PORT}"
            f"&hlink={PRIMARY_CONTEXT}&my_area={AREA}",
            json=ids,
        )
        self.assertEqual(2, deleted["count"])

    def test_03_patch_variants_validate_exists_true_and_false(self):
        entity_type = self.unique("ComdexPatchVariants")
        entity_id = f"urn:ngsi-ld:{entity_type}:001"
        self.addCleanup(self.delete_entity_best_effort, entity_id, BROKER1_PORT)

        self.post_entity(self.entity_payload(entity_type, entity_id), BROKER1_PORT)

        patched_full_checked = self.request_json(
            "PATCH",
            f"/ngsi-ld/v1/entities/{quote(entity_id, safe='')}/attrs"
            f"?broker={BROKER_HOST}&port={BROKER1_PORT}&qos={QOS}"
            f"&hlink={PRIMARY_CONTEXT}&my_area={AREA}&validate_exists=true",
            json={"language": {"type": "Property", "value": "FR"}},
        )
        self.assertEqual("updated", patched_full_checked["status"])

        patched_attr_checked = self.request_json(
            "PATCH",
            f"/ngsi-ld/v1/entities/{quote(entity_id, safe='')}/attrs/agencyName"
            f"?broker={BROKER_HOST}&port={BROKER1_PORT}&qos={QOS}"
            f"&hlink={PRIMARY_CONTEXT}&my_area={AREA}&validate_exists=true",
            json={"type": "Property", "value": "Checked attr patch"},
        )
        self.assertEqual("agencyName", patched_attr_checked["attr"])

        patched_full_fast = self.request_json(
            "PATCH",
            f"/ngsi-ld/v1/entities/{quote(entity_id, safe='')}/attrs"
            f"?broker={BROKER_HOST}&port={BROKER1_PORT}&qos={QOS}"
            f"&hlink={PRIMARY_CONTEXT}&my_area={AREA}&validate_exists=false"
            f"&entity_type={entity_type}",
            json={"language": {"type": "Property", "value": "ES"}},
        )
        self.assertEqual("updated", patched_full_fast["status"])

        patched_attr_fast = self.request_json(
            "PATCH",
            f"/ngsi-ld/v1/entities/{quote(entity_id, safe='')}/attrs/agencyName"
            f"?broker={BROKER_HOST}&port={BROKER1_PORT}&qos={QOS}"
            f"&hlink={PRIMARY_CONTEXT}&my_area={AREA}&validate_exists=false"
            f"&entity_type={entity_type}",
            json={"type": "Property", "value": "Fast attr patch"},
        )
        self.assertEqual("agencyName", patched_attr_fast["attr"])

        entities = self.get_entities_with_retry(
            f"/ngsi-ld/v1/entities?type={entity_type}&broker={BROKER_HOST}"
            f"&port={BROKER1_PORT}&hlink={PRIMARY_CONTEXT}",
        )
        matching_entities = [entity for entity in entities if entity.get("id") == entity_id]
        self.assertEqual(1, len(matching_entities))
        self.assertEqual("ES", matching_entities[0]["language"]["value"])
        self.assertEqual("Fast attr patch", matching_entities[0]["agencyName"]["value"])

        invalid_fast_patch = self.session.patch(
            f"{BASE_URL}/ngsi-ld/v1/entities/{quote(entity_id, safe='')}/attrs"
            f"?broker={BROKER_HOST}&port={BROKER1_PORT}&qos={QOS}"
            f"&hlink={PRIMARY_CONTEXT}&my_area={AREA}&validate_exists=false",
            json={"language": {"type": "Property", "value": "SHOULD_FAIL"}},
            timeout=15,
        )
        self.assertEqual(404, invalid_fast_patch.status_code)

    def test_04_subscription_websocket_receives_post_and_patch(self):
        asyncio.run(self._subscription_websocket_receives_post_and_patch())

    async def _subscription_websocket_receives_post_and_patch(self):
        entity_type = self.unique("ComdexSubscription")
        entity_id = f"urn:ngsi-ld:{entity_type}:001"
        subscription_id = f"urn:subscription:{entity_type}"
        self.addCleanup(self.delete_entity_best_effort, entity_id, BROKER2_PORT)
        self.addCleanup(self.delete_subscription_best_effort, subscription_id)

        self.request_json(
            "POST",
            f"/ngsi-ld/v1/subscriptions?broker={BROKER_HOST}&port={BROKER2_PORT}"
            f"&qos={QOS}&my_area={AREA}",
            expected_status=201,
            json={
                "id": subscription_id,
                "type": "Subscription",
                "entities": [{"type": entity_type}],
                "watchedAttributes": ["agencyName", "language"],
                "@context": CONTEXT,
            },
        )

        uri = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/{quote(subscription_id, safe='')}/ws"
        async with websockets.connect(uri) as websocket:
            _, connected = await self.recv_json(websocket, timeout=3)
            self.assertEqual("connected", connected["status"])

            self.post_entity(self.entity_payload(entity_type, entity_id), BROKER2_PORT)
            _, post_message = await self.wait_for_entity_message(
                websocket,
                entity_id,
                predicate=lambda msg: "agencyName" in msg or "language" in msg,
            )
            self.assertEqual(entity_id, post_message["id"])

            marker = "PATCH_DELIVERED"
            self.request_json(
                "PATCH",
                f"/ngsi-ld/v1/entities/{quote(entity_id, safe='')}/attrs"
                f"?broker={BROKER_HOST}&port={BROKER2_PORT}&qos={QOS}"
                f"&hlink={PRIMARY_CONTEXT}&my_area={AREA}",
                json={"agencyName": {"type": "Property", "value": marker}},
            )
            _, patch_message = await self.wait_for_entity_message(
                websocket,
                entity_id,
                predicate=lambda msg: msg.get("agencyName", {}).get("value") == marker,
            )
            self.assertEqual(marker, patch_message["agencyName"]["value"])

    def test_05_subscription_id_without_watched_attrs_and_list_get(self):
        asyncio.run(self._subscription_id_without_watched_attrs_and_list_get())

    async def _subscription_id_without_watched_attrs_and_list_get(self):
        entity_type = self.unique("ComdexIdSubscription")
        entity_id = f"urn:ngsi-ld:{entity_type}:001"
        subscription_id = f"urn:subscription:{entity_type}"
        self.addCleanup(self.delete_entity_best_effort, entity_id, BROKER2_PORT)
        self.addCleanup(self.delete_subscription_best_effort, subscription_id)

        created = self.request_json(
            "POST",
            f"/ngsi-ld/v1/subscriptions?broker={BROKER_HOST}&port={BROKER2_PORT}"
            f"&qos={QOS}&my_area={AREA}",
            expected_status=201,
            json={
                "id": subscription_id,
                "type": "Subscription",
                "entities": [{"id": entity_id}],
                "@context": CONTEXT,
            },
        )
        self.assertEqual(subscription_id, created["id"])

        subscriptions = self.request_json("GET", "/ngsi-ld/v1/subscriptions")
        listed = [subscription for subscription in subscriptions if subscription["id"] == subscription_id]
        self.assertEqual(1, len(listed))
        self.assertEqual(entity_id, listed[0]["entity_id"])

        fetched = self.request_json(
            "GET",
            f"/ngsi-ld/v1/subscriptions/{quote(subscription_id, safe='')}",
        )
        self.assertEqual(subscription_id, fetched["id"])
        self.assertEqual(entity_id, fetched["entity_id"])

        uri = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/{quote(subscription_id, safe='')}/ws"
        async with websockets.connect(uri) as websocket:
            _, connected = await self.recv_json(websocket, timeout=3)
            self.assertEqual("connected", connected["status"])

            self.post_entity(self.entity_payload(entity_type, entity_id), BROKER2_PORT)
            seen_attrs = set()
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline and seen_attrs != {"agencyName", "language"}:
                _, message = await self.wait_for_entity_message(
                    websocket,
                    entity_id,
                    timeout=deadline - time.monotonic(),
                )
                if "agencyName" in message:
                    seen_attrs.add("agencyName")
                if "language" in message:
                    seen_attrs.add("language")
            self.assertEqual({"agencyName", "language"}, seen_attrs)

    def test_06_provider_child_delete_stops_future_delivery(self):
        asyncio.run(self._provider_child_delete_stops_future_delivery())

    async def _provider_child_delete_stops_future_delivery(self):
        entity_type = self.unique("ComdexProviderStop")
        entity_id = f"urn:ngsi-ld:{entity_type}:001"
        subscription_id = f"urn:subscription:{entity_type}"
        self.addCleanup(self.delete_entity_best_effort, entity_id, BROKER2_PORT)
        self.addCleanup(self.delete_subscription_best_effort, subscription_id)

        self.request_json(
            "POST",
            f"/ngsi-ld/v1/subscriptions?broker={BROKER_HOST}&port={BROKER2_PORT}"
            f"&qos={QOS}&my_area={AREA}",
            expected_status=201,
            json={
                "id": subscription_id,
                "type": "Subscription",
                "entities": [{"type": entity_type}],
                "watchedAttributes": ["agencyName", "language"],
                "@context": CONTEXT,
            },
        )

        uri = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/{quote(subscription_id, safe='')}/ws"
        async with websockets.connect(uri) as websocket:
            await self.recv_json(websocket, timeout=3)
            self.post_entity(self.entity_payload(entity_type, entity_id), BROKER2_PORT)

            seen_attrs = set()
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline and seen_attrs != {"agencyName", "language"}:
                _, message = await self.wait_for_entity_message(websocket, entity_id, timeout=deadline - time.monotonic())
                if "agencyName" in message:
                    seen_attrs.add("agencyName")
                if "language" in message:
                    seen_attrs.add("language")
            self.assertEqual({"agencyName", "language"}, seen_attrs)

            providers = []
            for _ in range(20):
                providers = self.request_json(
                    "GET",
                    f"/ngsi-ld/v1/subscriptions/{quote(subscription_id, safe='')}/providers",
                )
                if providers:
                    break
                await asyncio.sleep(0.25)
            self.assertTrue(providers, "No provider child process was registered")

            provider = providers[0]
            stopped = self.request_json(
                "DELETE",
                f"/ngsi-ld/v1/subscriptions/{quote(subscription_id, safe='')}/providers"
                f"?provider_broker={provider['broker']}&provider_port={provider['port']}"
                f"&entity_type={entity_type}",
            )
            self.assertEqual("provider_stopped", stopped["status"])
            self.assertEqual([], stopped["remaining_providers"])

            marker = "AFTER_PROVIDER_CHILD_DELETE"
            self.request_json(
                "PATCH",
                f"/ngsi-ld/v1/entities/{quote(entity_id, safe='')}/attrs"
                f"?broker={BROKER_HOST}&port={BROKER2_PORT}&qos={QOS}"
                f"&hlink={PRIMARY_CONTEXT}&my_area={AREA}",
                json={"agencyName": {"type": "Property", "value": marker}},
            )

            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(websocket.recv(), timeout=4)

    def test_07_parent_subscription_stop_closes_websocket_cleanly(self):
        asyncio.run(self._parent_subscription_stop_closes_websocket_cleanly())

    async def _parent_subscription_stop_closes_websocket_cleanly(self):
        entity_type = self.unique("ComdexParentStop")
        subscription_id = f"urn:subscription:{entity_type}"
        self.addCleanup(self.delete_subscription_best_effort, subscription_id)

        self.request_json(
            "POST",
            f"/ngsi-ld/v1/subscriptions?broker={BROKER_HOST}&port={BROKER2_PORT}"
            f"&qos={QOS}&my_area={AREA}",
            expected_status=201,
            json={
                "id": subscription_id,
                "type": "Subscription",
                "entities": [{"type": entity_type}],
                "@context": CONTEXT,
            },
        )

        uri = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/{quote(subscription_id, safe='')}/ws"
        async with websockets.connect(uri) as websocket:
            _, connected = await self.recv_json(websocket, timeout=3)
            self.assertEqual("connected", connected["status"])

            stopped = self.request_json(
                "DELETE",
                f"/ngsi-ld/v1/subscriptions/{quote(subscription_id, safe='')}",
            )
            self.assertEqual("stopped", stopped["status"])

            _, stop_message = await self.recv_json(websocket, timeout=5)
            self.assertEqual({"status": "stopped", "id": subscription_id}, stop_message)


    # ------------------------------------------------------------------
    # test_08 — batch create
    # ------------------------------------------------------------------

    def test_08_batch_create(self):
        """POST /entityOperations/create stores all entities and they are retrievable.

        NOTE: batch_create uses post_entity with bypass_existence_check=0 for the first
        entity of each type.  That path calls check_existence() twice (entity + provider
        advertisement), each of which creates a temporary paho client.  If the auto-
        generated client ID collides with the batch client's own ID, the broker evicts
        the batch client and retained publishes are silently lost.  The retry loop below
        gives MQTT time to settle; if all retries return empty the test still fails,
        which flags the underlying batch_create reliability bug.
        """
        entity_type = self.unique("ComdexBatchCreate")
        ids = [f"urn:ngsi-ld:{entity_type}:{i}" for i in range(3)]
        for entity_id in ids:
            self.addCleanup(self.delete_entity_best_effort, entity_id, BROKER1_PORT)

        entities = [
            self.entity_payload(entity_type, ids[0], name="Create 1", language="EN"),
            self.entity_payload(entity_type, ids[1], name="Create 2", language="DE"),
            self.entity_payload(entity_type, ids[2], name="Create 3", language="FR"),
        ]
        result = self.request_json(
            "POST",
            f"/ngsi-ld/v1/entityOperations/create?broker={BROKER_HOST}&port={BROKER1_PORT}"
            f"&qos={QOS}&my_area={AREA}",
            json=entities,
        )
        self.assertEqual("created", result["status"])
        self.assertEqual(3, result["count"])

        fetched = self.get_entities_with_retry(
            f"/ngsi-ld/v1/entities?type={entity_type}&broker={BROKER_HOST}"
            f"&port={BROKER1_PORT}&hlink={PRIMARY_CONTEXT}",
        )
        fetched_ids = {e["id"] for e in fetched}
        self.assertTrue(set(ids).issubset(fetched_ids), f"Expected {ids} in {fetched_ids}")

    # ------------------------------------------------------------------
    # test_09 — GET entities filtered by id
    # ------------------------------------------------------------------

    def test_09_get_entities_by_id_filter(self):
        """GET /entities?id=X returns only the entity with that specific ID."""
        entity_type = self.unique("ComdexIdFilter")
        id_a = f"urn:ngsi-ld:{entity_type}:aaa"
        id_b = f"urn:ngsi-ld:{entity_type}:bbb"
        self.addCleanup(self.delete_entity_best_effort, id_a, BROKER1_PORT)
        self.addCleanup(self.delete_entity_best_effort, id_b, BROKER1_PORT)

        self.post_entity(self.entity_payload(entity_type, id_a, name="Agency A"), BROKER1_PORT)
        self.post_entity(self.entity_payload(entity_type, id_b, name="Agency B"), BROKER1_PORT)

        result = self.request_json(
            "GET",
            f"/ngsi-ld/v1/entities?id={quote(id_a, safe='')}&broker={BROKER_HOST}"
            f"&port={BROKER1_PORT}&hlink={PRIMARY_CONTEXT}",
        )
        result_ids = {e["id"] for e in result}
        self.assertIn(id_a, result_ids, "Filtered entity should be returned")
        self.assertNotIn(id_b, result_ids, "Non-matching entity must not appear in id-filtered result")

    # ------------------------------------------------------------------
    # test_10 — GET entities with attribute projection
    # ------------------------------------------------------------------

    def test_10_get_entities_with_attrs_projection(self):
        """GET /entities?attrs=agencyName returns entity with only the requested attribute."""
        entity_type = self.unique("ComdexAttrsProjection")
        entity_id = f"urn:ngsi-ld:{entity_type}:001"
        self.addCleanup(self.delete_entity_best_effort, entity_id, BROKER1_PORT)

        self.post_entity(self.entity_payload(entity_type, entity_id), BROKER1_PORT)

        result = self.get_entities_with_retry(
            f"/ngsi-ld/v1/entities?type={entity_type}&attrs=agencyName"
            f"&broker={BROKER_HOST}&port={BROKER1_PORT}&hlink={PRIMARY_CONTEXT}",
        )
        matching = [e for e in result if e.get("id") == entity_id]
        self.assertEqual(1, len(matching), "Entity must be retrievable after POST")
        entity = matching[0]
        self.assertIn("agencyName", entity, "Requested attribute must be present")
        self.assertNotIn("language", entity, "Non-requested attribute must be projected out")

    # ------------------------------------------------------------------
    # test_11 — GET entities with limit
    # ------------------------------------------------------------------

    def test_11_get_entities_with_limit(self):
        """GET /entities?limit=N returns at most N entities."""
        entity_type = self.unique("ComdexLimit")
        ids = [f"urn:ngsi-ld:{entity_type}:{i}" for i in range(4)]
        for entity_id in ids:
            self.addCleanup(self.delete_entity_best_effort, entity_id, BROKER1_PORT)
            self.post_entity(self.entity_payload(entity_type, entity_id), BROKER1_PORT)

        result = self.request_json(
            "GET",
            f"/ngsi-ld/v1/entities?type={entity_type}&limit=2"
            f"&broker={BROKER_HOST}&port={BROKER1_PORT}&hlink={PRIMARY_CONTEXT}",
        )
        self.assertLessEqual(len(result), 2, "Result must not exceed the requested limit")

    # ------------------------------------------------------------------
    # test_12 — subscription with combined type + id filter
    # ------------------------------------------------------------------

    def test_12_subscription_type_and_id_combined_filter(self):
        asyncio.run(self._subscription_type_and_id_combined_filter())

    async def _subscription_type_and_id_combined_filter(self):
        """Subscription with both type and id in entities[] delivers only the target entity."""
        entity_type = self.unique("ComdexTypeIdFilter")
        target_id   = f"urn:ngsi-ld:{entity_type}:target"
        other_id    = f"urn:ngsi-ld:{entity_type}:other"
        sub_id      = f"urn:subscription:{entity_type}"
        self.addCleanup(self.delete_entity_best_effort, target_id, BROKER2_PORT)
        self.addCleanup(self.delete_entity_best_effort, other_id,  BROKER2_PORT)
        self.addCleanup(self.delete_subscription_best_effort, sub_id)

        self.request_json(
            "POST",
            f"/ngsi-ld/v1/subscriptions?broker={BROKER_HOST}&port={BROKER2_PORT}"
            f"&qos={QOS}&my_area={AREA}",
            expected_status=201,
            json={
                "id": sub_id,
                "type": "Subscription",
                "entities": [{"type": entity_type, "id": target_id}],
                "@context": CONTEXT,
            },
        )

        uri = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/{quote(sub_id, safe='')}/ws"
        async with websockets.connect(uri) as ws:
            _, connected = await self.recv_json(ws, timeout=3)
            self.assertEqual("connected", connected["status"])

            # Post a different entity of the same type first — should NOT be delivered.
            self.post_entity(self.entity_payload(entity_type, other_id, name="Other"), BROKER2_PORT)
            # Post the target entity — SHOULD be delivered.
            self.post_entity(self.entity_payload(entity_type, target_id, name="Target"), BROKER2_PORT)

            # Collect all messages received in the next 6 seconds.
            seen_ids = set()
            deadline = time.monotonic() + 6
            try:
                while time.monotonic() < deadline:
                    remaining = max(0.05, deadline - time.monotonic())
                    _, msg = await self.recv_json(ws, timeout=remaining)
                    if isinstance(msg, dict) and msg.get("id") not in (None, sub_id):
                        seen_ids.add(msg["id"])
            except asyncio.TimeoutError:
                pass

            self.assertIn(target_id, seen_ids, "Target entity must be delivered")
            self.assertNotIn(other_id, seen_ids, "Non-matching entity must NOT be delivered")

    # ------------------------------------------------------------------
    # test_13 — watchedAttributes filters out non-watched patches
    # ------------------------------------------------------------------

    def test_13_watched_attributes_filters_non_watched_patch(self):
        asyncio.run(self._watched_attributes_filters_non_watched_patch())

    async def _watched_attributes_filters_non_watched_patch(self):
        """A PATCH that only touches attributes not in watchedAttributes must not trigger a notification."""
        entity_type = self.unique("ComdexWatchedFilter")
        entity_id   = f"urn:ngsi-ld:{entity_type}:001"
        sub_id      = f"urn:subscription:{entity_type}"
        self.addCleanup(self.delete_entity_best_effort, entity_id, BROKER2_PORT)
        self.addCleanup(self.delete_subscription_best_effort, sub_id)

        # Subscription watches agencyName only — language changes must be silent.
        self.request_json(
            "POST",
            f"/ngsi-ld/v1/subscriptions?broker={BROKER_HOST}&port={BROKER2_PORT}"
            f"&qos={QOS}&my_area={AREA}",
            expected_status=201,
            json={
                "id": sub_id,
                "type": "Subscription",
                "entities": [{"type": entity_type}],
                "watchedAttributes": ["agencyName"],
                "@context": CONTEXT,
            },
        )

        uri = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/{quote(sub_id, safe='')}/ws"
        async with websockets.connect(uri) as ws:
            _, connected = await self.recv_json(ws, timeout=3)
            self.assertEqual("connected", connected["status"])

            # POST entity and drain the initial notification (snapshot or live).
            self.post_entity(self.entity_payload(entity_type, entity_id), BROKER2_PORT)
            await self.wait_for_entity_message(
                ws, entity_id, timeout=8,
                predicate=lambda msg: "agencyName" in msg,
            )

            # PATCH only language — not in watchedAttributes.
            self.request_json(
                "PATCH",
                f"/ngsi-ld/v1/entities/{quote(entity_id, safe='')}/attrs"
                f"?broker={BROKER_HOST}&port={BROKER2_PORT}&qos={QOS}"
                f"&hlink={PRIMARY_CONTEXT}&my_area={AREA}",
                json={"language": {"type": "Property", "value": "PT"}},
            )

            # No notification should arrive for the next 4 seconds.
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(ws.recv(), timeout=4)

    # ------------------------------------------------------------------
    # test_14 — unknown subscription → 404 on all read/write endpoints
    # ------------------------------------------------------------------

    def test_14_unknown_subscription_returns_404(self):
        fake_id = f"urn:subscription:nonexistent:{uuid.uuid4().hex}"

        self.request_json(
            "GET",
            f"/ngsi-ld/v1/subscriptions/{quote(fake_id, safe='')}",
            expected_status=404,
        )
        self.request_json(
            "DELETE",
            f"/ngsi-ld/v1/subscriptions/{quote(fake_id, safe='')}",
            expected_status=404,
        )
        self.request_json(
            "GET",
            f"/ngsi-ld/v1/subscriptions/{quote(fake_id, safe='')}/providers",
            expected_status=404,
        )

    # ------------------------------------------------------------------
    # test_15 — DELETE /providers with no filter params → 400
    # ------------------------------------------------------------------

    def test_15_provider_delete_without_filter_returns_400(self):
        entity_type = self.unique("ComdexProviderNoFilter")
        sub_id = f"urn:subscription:{entity_type}"
        self.addCleanup(self.delete_subscription_best_effort, sub_id)

        self.request_json(
            "POST",
            f"/ngsi-ld/v1/subscriptions?broker={BROKER_HOST}&port={BROKER2_PORT}"
            f"&qos={QOS}&my_area={AREA}",
            expected_status=201,
            json={
                "id": sub_id,
                "type": "Subscription",
                "entities": [{"type": entity_type}],
                "@context": CONTEXT,
            },
        )

        # No query params at all — server must reject with 400.
        response = self.session.delete(
            f"{BASE_URL}/ngsi-ld/v1/subscriptions/{quote(sub_id, safe='')}/providers",
            timeout=15,
        )
        self.assertEqual(400, response.status_code)

    # ------------------------------------------------------------------
    # test_16 — WS connect to unknown subscription → server closes 1008
    # ------------------------------------------------------------------

    def test_16_ws_unknown_subscription_closes_with_error(self):
        asyncio.run(self._ws_unknown_subscription_closes_with_error())

    async def _ws_unknown_subscription_closes_with_error(self):
        """Connecting to the WS of a non-existent subscription must result in a closed connection.

        FastAPI closes the WS before accepting it (code=1008), which the websockets library
        surfaces as InvalidStatus (HTTP 403) rather than a post-handshake close frame.
        """
        fake_id = f"urn:subscription:nonexistent:{uuid.uuid4().hex}"
        uri = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/{quote(fake_id, safe='')}/ws"

        with self.assertRaises(
            (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.InvalidStatus,
            )
        ):
            async with websockets.connect(uri) as ws:
                await ws.recv()

    # ------------------------------------------------------------------
    # test_17 — one-shot WS endpoint: create + stream in a single connection
    # ------------------------------------------------------------------

    def test_17_ws_create_and_stream_in_one_connection(self):
        asyncio.run(self._ws_create_and_stream_in_one_connection())

    async def _ws_create_and_stream_in_one_connection(self):
        """WS /subscriptions/ws creates a subscription inline and streams notifications.

        A WebSocket disconnect is a transport failure, not a delete request:
        the logical subscription must survive it (see test_34+ for the full
        create-or-resume lifecycle). Only explicit DELETE removes it here.
        """
        entity_type = self.unique("ComdexWsStream")
        entity_id   = f"urn:ngsi-ld:{entity_type}:001"
        sub_id      = None
        self.addCleanup(self.delete_entity_best_effort, entity_id, BROKER2_PORT)

        uri = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/ws"
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({
                "id": f"urn:subscription:{entity_type}",
                "type": "Subscription",
                "entities": [{"type": entity_type}],
                "@context": CONTEXT,
                "broker": BROKER_HOST,
                "port":   BROKER2_PORT,
                "qos":    QOS,
            }))

            _, subscribed = await self.recv_json(ws, timeout=5)
            self.assertEqual("subscribed", subscribed["status"])
            sub_id = subscribed["id"]
            self.addCleanup(self.delete_subscription_best_effort, sub_id)

            self.post_entity(self.entity_payload(entity_type, entity_id), BROKER2_PORT)
            _, msg = await self.wait_for_entity_message(ws, entity_id, timeout=8)
            self.assertEqual(entity_id, msg["id"])
        # The async-with block exits here: client sends a close frame.

        await asyncio.sleep(1)
        subs = self.request_json("GET", "/ngsi-ld/v1/subscriptions")
        active_ids = {s["id"] for s in subs}
        self.assertIn(sub_id, active_ids, "WebSocket disconnect must not delete the subscription")

        self.session.delete(f"{BASE_URL}/ngsi-ld/v1/subscriptions/{quote(sub_id, safe='')}", timeout=10)
        subs = self.request_json("GET", "/ngsi-ld/v1/subscriptions")
        self.assertNotIn(sub_id, {s["id"] for s in subs}, "Explicit DELETE must still remove the subscription")

    # ------------------------------------------------------------------
    # test_18 — WS client disconnect leaves subscription alive; reconnect works
    # ------------------------------------------------------------------

    def test_18_ws_client_reconnect_after_disconnect(self):
        asyncio.run(self._ws_client_reconnect_after_disconnect())

    async def _ws_client_reconnect_after_disconnect(self):
        """A subscription must survive a WS client disconnect and accept a new connection."""
        entity_type = self.unique("ComdexReconnect")
        entity_id   = f"urn:ngsi-ld:{entity_type}:001"
        sub_id      = f"urn:subscription:{entity_type}"
        self.addCleanup(self.delete_entity_best_effort, entity_id, BROKER2_PORT)
        self.addCleanup(self.delete_subscription_best_effort, sub_id)

        self.request_json(
            "POST",
            f"/ngsi-ld/v1/subscriptions?broker={BROKER_HOST}&port={BROKER2_PORT}"
            f"&qos={QOS}&my_area={AREA}",
            expected_status=201,
            json={
                "id": sub_id,
                "type": "Subscription",
                "entities": [{"type": entity_type}],
                "@context": CONTEXT,
            },
        )

        uri = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/{quote(sub_id, safe='')}/ws"

        # First client connects then immediately disconnects.
        ws1 = await websockets.connect(uri)
        _, connected = await self.recv_json(ws1, timeout=3)
        self.assertEqual("connected", connected["status"])
        await ws1.close()
        await asyncio.sleep(0.3)

        # Subscription must still be active.
        subs = self.request_json("GET", "/ngsi-ld/v1/subscriptions")
        active_ids = {s["id"] for s in subs}
        self.assertIn(sub_id, active_ids, "Subscription must survive WS client disconnect")

        # Second client reconnects and must receive notifications normally.
        async with websockets.connect(uri) as ws2:
            _, connected2 = await self.recv_json(ws2, timeout=3)
            self.assertEqual("connected", connected2["status"])

            self.post_entity(self.entity_payload(entity_type, entity_id), BROKER2_PORT)
            _, msg = await self.wait_for_entity_message(ws2, entity_id, timeout=8)
            self.assertEqual(entity_id, msg["id"])


    # ------------------------------------------------------------------
    # test_19-21 (q filter on subscriptions) removed: post_subscription()
    # in actionhandler.py never reads/stores a "q" field, and neither the
    # WS snapshot nor the live notification path apply it. Re-add these
    # once subscription-level q filtering is implemented.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # test_22 — subscription without q filter still works normally
    # ------------------------------------------------------------------

    def test_22_subscription_no_q_filter_still_works(self):
        asyncio.run(self._subscription_no_q_filter_still_works())

    async def _subscription_no_q_filter_still_works(self):
        """Omitting q must behave exactly as before (no regression)."""
        entity_type = self.unique("ComdexNoQ")
        entity_id   = f"urn:ngsi-ld:{entity_type}:001"
        sub_id      = f"urn:subscription:{entity_type}"

        self.addCleanup(self.delete_entity_best_effort, entity_id, BROKER2_PORT)
        self.addCleanup(self.delete_subscription_best_effort, sub_id)

        self.request_json(
            "POST",
            f"/ngsi-ld/v1/subscriptions?broker={BROKER_HOST}&port={BROKER2_PORT}"
            f"&qos={QOS}&my_area={AREA}",
            expected_status=201,
            json={
                "id": sub_id,
                "type": "Subscription",
                "entities": [{"type": entity_type}],
                "watchedAttributes": ["agencyName", "language"],
                "@context": CONTEXT,
            },
        )

        fetched = self.request_json(
            "GET", f"/ngsi-ld/v1/subscriptions/{quote(sub_id, safe='')}"
        )
        self.assertIsNone(fetched.get("q"), "q must be null when not set")

        uri = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/{quote(sub_id, safe='')}/ws"
        async with websockets.connect(uri) as ws:
            _, connected = await self.recv_json(ws, timeout=3)
            self.assertEqual("connected", connected["status"])

            # Both EN and FR entities should be delivered (no q filter)
            for lang in ("EN", "FR"):
                eid = f"urn:ngsi-ld:{entity_type}:{lang}"
                self.addCleanup(self.delete_entity_best_effort, eid, BROKER2_PORT)
                self.post_entity(
                    self.entity_payload(entity_type, eid, language=lang),
                    BROKER2_PORT,
                )

            received_ids: set[str] = set()
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and len(received_ids) < 2:
                try:
                    remaining = max(0.1, deadline - time.monotonic())
                    _, msg = await self.recv_json(ws, timeout=remaining)
                    if (
                        isinstance(msg, dict)
                        and isinstance(msg.get("id"), str)
                        and msg["id"].startswith(f"urn:ngsi-ld:{entity_type}:")
                    ):
                        received_ids.add(msg["id"])
                except asyncio.TimeoutError:
                    break

            self.assertEqual(2, len(received_ids),
                             "Without q filter, all entity types must be delivered")

    # ------------------------------------------------------------------
    # test_23 — one-shot WS subscription must survive the idle-grace reaper
    # while its WebSocket stays connected.
    #
    # Regression for: POST /ngsi-ld/v1/subscriptions/ws registered a
    # subscription as ws_connected=False (disconnected_at=now), the same
    # idle state used by the reconnectable endpoint, so the abandoned-
    # subscription reaper removed the subscription (and its provider child
    # processes) after SUBSCRIPTION_IDLE_GRACE_SECONDS even though the
    # client's WebSocket was still open and actively receiving.
    # ------------------------------------------------------------------

    def test_23_ws_one_shot_subscription_survives_idle_grace_while_connected(self):
        asyncio.run(self._ws_one_shot_subscription_survives_idle_grace_while_connected())

    async def _ws_one_shot_subscription_survives_idle_grace_while_connected(self):
        """A live one-shot WS subscription must not be reaped as abandoned."""
        grace_seconds = 3
        port = 8010
        env = dict(os.environ)
        env["COMDEX_SUBSCRIPTION_IDLE_GRACE_SECONDS"] = str(grace_seconds)
        env["COMDEX_SUBSCRIPTION_REAPER_POLL_SECONDS"] = "1"

        root = Path(__file__).resolve().parents[1]
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "actionhandlerAPI:app",
                "--host", "127.0.0.1", "--port", str(port),
            ],
            cwd=root, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.addCleanup(self._terminate_process, proc)
        try:
            self.assertTrue(
                wait_for_api(timeout=15.0, base_url=f"http://127.0.0.1:{port}"),
                "Dedicated test server did not start",
            )

            base_url = f"http://127.0.0.1:{port}"
            ws_base_url = base_url.replace("http://", "ws://")
            entity_type = self.unique("ComdexIdleGrace")
            entity_id = f"urn:ngsi-ld:{entity_type}:001"
            self.addCleanup(self.delete_entity_best_effort, entity_id, BROKER2_PORT)

            uri = f"{ws_base_url}/ngsi-ld/v1/subscriptions/ws"
            async with websockets.connect(uri) as ws:
                await ws.send(json.dumps({
                    "id": f"urn:subscription:{entity_type}",
                    "type": "Subscription",
                    "entities": [{"type": entity_type}],
                    "@context": CONTEXT,
                    "broker": BROKER_HOST,
                    "port": BROKER2_PORT,
                    "qos": QOS,
                }))
                _, subscribed = await self.recv_json(ws, timeout=5)
                self.assertEqual("subscribed", subscribed["status"])
                sub_id = subscribed["id"]

                # Outlive the idle grace period (and several reaper polls)
                # while the WebSocket remains connected the whole time.
                await asyncio.sleep(grace_seconds * 3)

                response = requests.get(f"{base_url}/ngsi-ld/v1/subscriptions/{sub_id}", timeout=5)
                self.assertEqual(
                    200, response.status_code,
                    "Live subscription must not be reaped while its WebSocket is connected",
                )

                # It must still be functional, not just present.
                self.post_entity(self.entity_payload(entity_type, entity_id), BROKER2_PORT)
                _, msg = await self.wait_for_entity_message(ws, entity_id, timeout=8)
                self.assertEqual(entity_id, msg["id"])
        finally:
            self._terminate_process(proc)

    @staticmethod
    def _terminate_process(proc):
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    # ------------------------------------------------------------------
    # Deletion-engine regression tests (test_24+)
    #
    # These exercise the refactored delete_entity()/batch_delete() engine
    # in actionhandler.py. Retained-topic presence is checked directly over
    # MQTT (not through the ComDeX API) because there is no HTTP endpoint
    # that lists provider advertisements — this is test-only introspection
    # of internal state, unrelated to the "use the API, not raw MQTT" rule
    # that applies to the benchmark harness.
    # ------------------------------------------------------------------

    def mqtt_topic_exists(self, broker, port, topic_filter, timeout=2.0):
        found = []

        def on_message(client, userdata, msg):
            if msg.retain:
                found.append(msg.topic)

        client = mqtt.Client()
        client.on_message = on_message
        client.connect(broker, port)
        client.loop_start()
        client.subscribe(topic_filter, qos=1)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not found:
            time.sleep(0.05)
        client.loop_stop()
        client.disconnect()
        return bool(found)

    def advertisement_topic(self, broker, port, area, context, entity_type, entity_id=None):
        hlink = context.replace("/", "§")
        base = f"provider/{broker}/{port}/{area}/{hlink}/{entity_type}"
        return base if entity_id is None else f"{base}/{entity_id}"

    def entity_topic_filter(self, area, context, entity_type, entity_id):
        hlink = context.replace("/", "§")
        return f"{area}/entities/{hlink}/{entity_type}/+/{entity_id}/#"

    def test_24_single_delete_removes_topics_and_advertisement(self):
        """Deleting the only entity of a type must clear its attribute topics and the advertisement."""
        entity_type = self.unique("ComdexDelSingle")
        entity_id = f"urn:ngsi-ld:{entity_type}:001"
        self.post_entity(self.entity_payload(entity_type, entity_id), BROKER1_PORT)
        self.assertTrue(
            self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, self.entity_topic_filter(AREA, PRIMARY_CONTEXT, entity_type, entity_id)),
            "Entity must exist right after creation",
        )
        self.request_json(
            "DELETE",
            f"/ngsi-ld/v1/entities/{quote(entity_id, safe='')}"
            f"?broker={BROKER_HOST}&port={BROKER1_PORT}&hlink={PRIMARY_CONTEXT}&my_area={AREA}",
        )
        self.assertFalse(
            self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, self.entity_topic_filter(AREA, PRIMARY_CONTEXT, entity_type, entity_id)),
            "Entity attribute topics must be gone after delete",
        )
        self.assertFalse(
            self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, self.advertisement_topic(BROKER_HOST, BROKER1_PORT, AREA, PRIMARY_CONTEXT, entity_type)),
            "Provider advertisement must be removed once the last entity of that type is gone",
        )

    def test_25_single_delete_keeps_advertisement_when_sibling_remains(self):
        """Deleting one entity must not remove the advertisement while a sibling of the same type exists."""
        entity_type = self.unique("ComdexDelSibling")
        id_a = f"urn:ngsi-ld:{entity_type}:A"
        id_b = f"urn:ngsi-ld:{entity_type}:B"
        self.addCleanup(self.delete_entity_best_effort, id_b, BROKER1_PORT)
        self.post_entity(self.entity_payload(entity_type, id_a), BROKER1_PORT)
        self.post_entity(self.entity_payload(entity_type, id_b), BROKER1_PORT)

        self.request_json(
            "DELETE",
            f"/ngsi-ld/v1/entities/{quote(id_a, safe='')}"
            f"?broker={BROKER_HOST}&port={BROKER1_PORT}&hlink={PRIMARY_CONTEXT}&my_area={AREA}",
        )
        self.assertFalse(
            self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, self.entity_topic_filter(AREA, PRIMARY_CONTEXT, entity_type, id_a)),
            "Deleted entity's own topics must be gone",
        )
        self.assertTrue(
            self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, self.entity_topic_filter(AREA, PRIMARY_CONTEXT, entity_type, id_b)),
            "Sibling entity of the same type must remain",
        )
        self.assertTrue(
            self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, self.advertisement_topic(BROKER_HOST, BROKER1_PORT, AREA, PRIMARY_CONTEXT, entity_type)),
            "Advertisement must remain while a sibling entity of that type still exists",
        )

    def test_26_delete_nonexistent_entity_returns_404(self):
        entity_id = self.unique("urn:ngsi-ld:ComdexDelMissing:")
        self.request_json(
            "DELETE",
            f"/ngsi-ld/v1/entities/{quote(entity_id, safe='')}"
            f"?broker={BROKER_HOST}&port={BROKER1_PORT}&hlink={PRIMARY_CONTEXT}&my_area={AREA}",
            expected_status=404,
        )

    def test_27_batch_delete_same_type_removes_all_and_advertisement(self):
        entity_type = self.unique("ComdexBatchSame")
        ids = [f"urn:ngsi-ld:{entity_type}:{i:03}" for i in range(12)]
        for eid in ids:
            self.post_entity(self.entity_payload(entity_type, eid), BROKER1_PORT)

        self.request_json(
            "POST",
            f"/ngsi-ld/v1/entityOperations/delete?broker={BROKER_HOST}&port={BROKER1_PORT}"
            f"&hlink={PRIMARY_CONTEXT}&my_area={AREA}",
            json=ids,
        )
        for eid in ids:
            self.assertFalse(
                self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, self.entity_topic_filter(AREA, PRIMARY_CONTEXT, entity_type, eid)),
                f"{eid} must be gone after batch delete",
            )
        self.assertFalse(
            self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, self.advertisement_topic(BROKER_HOST, BROKER1_PORT, AREA, PRIMARY_CONTEXT, entity_type)),
            "Advertisement must be removed once every entity of that type is gone",
        )

    def test_28_batch_delete_partial_type_keeps_remaining_and_advertisement(self):
        entity_type = self.unique("ComdexBatchPartial")
        ids = [f"urn:ngsi-ld:{entity_type}:{i:03}" for i in range(20)]
        to_delete, to_keep = ids[:10], ids[10:]
        for eid in ids:
            self.post_entity(self.entity_payload(entity_type, eid), BROKER1_PORT)
        self.addCleanup(lambda: [self.delete_entity_best_effort(e, BROKER1_PORT) for e in to_keep])

        self.request_json(
            "POST",
            f"/ngsi-ld/v1/entityOperations/delete?broker={BROKER_HOST}&port={BROKER1_PORT}"
            f"&hlink={PRIMARY_CONTEXT}&my_area={AREA}",
            json=to_delete,
        )
        for eid in to_delete:
            self.assertFalse(self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, self.entity_topic_filter(AREA, PRIMARY_CONTEXT, entity_type, eid)))
        for eid in to_keep:
            self.assertTrue(self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, self.entity_topic_filter(AREA, PRIMARY_CONTEXT, entity_type, eid)),
                            f"{eid} must remain: only half the type's entities were deleted")
        self.assertTrue(
            self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, self.advertisement_topic(BROKER_HOST, BROKER1_PORT, AREA, PRIMARY_CONTEXT, entity_type)),
            "Advertisement must remain: entities of this type still exist",
        )

    def test_29_batch_delete_mixed_types_handled_independently(self):
        type_a = self.unique("ComdexMixedA")
        type_b = self.unique("ComdexMixedB")
        ids_a = [f"urn:ngsi-ld:{type_a}:{i}" for i in range(5)]
        ids_b = [f"urn:ngsi-ld:{type_b}:{i}" for i in range(5)]
        for eid in ids_a:
            self.post_entity(self.entity_payload(type_a, eid), BROKER1_PORT)
        for eid in ids_b:
            self.post_entity(self.entity_payload(type_b, eid), BROKER1_PORT)
        self.addCleanup(lambda: [self.delete_entity_best_effort(e, BROKER1_PORT) for e in ids_b])

        # Delete only type A entirely; type B must be completely untouched.
        self.request_json(
            "POST",
            f"/ngsi-ld/v1/entityOperations/delete?broker={BROKER_HOST}&port={BROKER1_PORT}"
            f"&hlink={PRIMARY_CONTEXT}&my_area={AREA}",
            json=ids_a,
        )
        self.assertFalse(
            self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, self.advertisement_topic(BROKER_HOST, BROKER1_PORT, AREA, PRIMARY_CONTEXT, type_a)),
            "Type A advertisement must be removed",
        )
        self.assertTrue(
            self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, self.advertisement_topic(BROKER_HOST, BROKER1_PORT, AREA, PRIMARY_CONTEXT, type_b)),
            "Type B advertisement must be unaffected by deleting all of type A "
            "(regression: the old wildcard advertisement-clear topic could wipe out other types)",
        )
        for eid in ids_b:
            self.assertTrue(self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, self.entity_topic_filter(AREA, PRIMARY_CONTEXT, type_b, eid)))

    def test_30_batch_delete_duplicate_ids_no_error(self):
        entity_type = self.unique("ComdexDup")
        entity_id = f"urn:ngsi-ld:{entity_type}:001"
        self.post_entity(self.entity_payload(entity_type, entity_id), BROKER1_PORT)

        self.request_json(
            "POST",
            f"/ngsi-ld/v1/entityOperations/delete?broker={BROKER_HOST}&port={BROKER1_PORT}"
            f"&hlink={PRIMARY_CONTEXT}&my_area={AREA}",
            json=[entity_id, entity_id, entity_id],
        )
        self.assertFalse(self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, self.entity_topic_filter(AREA, PRIMARY_CONTEXT, entity_type, entity_id)))

    def test_31_batch_delete_mixed_existing_and_missing_ids(self):
        entity_type = self.unique("ComdexMixedExist")
        existing = f"urn:ngsi-ld:{entity_type}:real"
        missing = f"urn:ngsi-ld:{entity_type}:ghost"
        self.post_entity(self.entity_payload(entity_type, existing), BROKER1_PORT)

        response = self.request_json(
            "POST",
            f"/ngsi-ld/v1/entityOperations/delete?broker={BROKER_HOST}&port={BROKER1_PORT}"
            f"&hlink={PRIMARY_CONTEXT}&my_area={AREA}",
            json=[existing, missing],
        )
        self.assertEqual(2, response["count"], "Batch response must count the input, not just what existed")
        self.assertFalse(self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, self.entity_topic_filter(AREA, PRIMARY_CONTEXT, entity_type, existing)))

    def test_32_delete_wildcard_hlink_finds_entity_created_with_explicit_context(self):
        entity_type = self.unique("ComdexWildcardHlink")
        entity_id = f"urn:ngsi-ld:{entity_type}:001"
        self.post_entity(self.entity_payload(entity_type, entity_id), BROKER1_PORT)

        self.request_json(
            "DELETE",
            f"/ngsi-ld/v1/entities/{quote(entity_id, safe='')}?broker={BROKER_HOST}&port={BROKER1_PORT}&my_area={AREA}",
        )
        self.assertFalse(self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, self.entity_topic_filter(AREA, PRIMARY_CONTEXT, entity_type, entity_id)))

    def test_33_singleidadvertisement_true_mode_scopes_advertisement_per_entity(self):
        """With singleidadvertisement=True, each entity has its own advertisement
        topic; deleting one entity must not touch a sibling's advertisement."""
        asyncio.run(self._singleidadvertisement_true_mode_scopes_advertisement_per_entity())

    async def _singleidadvertisement_true_mode_scopes_advertisement_per_entity(self):
        port = 8011
        env = dict(os.environ)
        env["COMDEX_SINGLE_ID_ADVERTISEMENT"] = "true"
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "actionhandlerAPI:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.addCleanup(self._terminate_process, proc)
        base_url = f"http://127.0.0.1:{port}"
        self.assertTrue(wait_for_api(timeout=15.0, base_url=base_url), "Dedicated singleidadvertisement=True server did not start")

        entity_type = self.unique("ComdexSingleIdAdvert")
        id_a = f"urn:ngsi-ld:{entity_type}:A"
        id_b = f"urn:ngsi-ld:{entity_type}:B"
        self.addCleanup(self.delete_entity_best_effort, id_b, BROKER1_PORT)
        for eid in (id_a, id_b):
            requests.post(
                f"{base_url}/ngsi-ld/v1/entities?broker={BROKER_HOST}&port={BROKER1_PORT}&qos={QOS}&my_area={AREA}",
                json=self.entity_payload(entity_type, eid), timeout=15,
            )

        advert_a = self.advertisement_topic(BROKER_HOST, BROKER1_PORT, AREA, PRIMARY_CONTEXT, entity_type, id_a)
        advert_b = self.advertisement_topic(BROKER_HOST, BROKER1_PORT, AREA, PRIMARY_CONTEXT, entity_type, id_b)
        self.assertTrue(self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, advert_a))
        self.assertTrue(self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, advert_b))

        r = requests.delete(
            f"{base_url}/ngsi-ld/v1/entities/{quote(id_a, safe='')}"
            f"?broker={BROKER_HOST}&port={BROKER1_PORT}&hlink={PRIMARY_CONTEXT}&my_area={AREA}",
            timeout=15,
        )
        self.assertEqual(200, r.status_code)
        self.assertFalse(self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, advert_a), "A's own advertisement must be gone")
        self.assertTrue(self.mqtt_topic_exists(BROKER_HOST, BROKER1_PORT, advert_b), "B's advertisement is independent and must remain")

    # ------------------------------------------------------------------
    # One-shot endpoint (WS /ngsi-ld/v1/subscriptions/ws) create-or-resume
    # lifecycle regression tests (test_34+).
    #
    # Regression for: a WebSocket disconnect (timeout, network blip, client
    # restart) used to call stop_subscription() directly, tearing down the
    # logical subscription and every provider child process along with the
    # transport. Under sustained backlog this meant a transient WS timeout
    # destroyed 40 live provider connections. Disconnect must now only mark
    # the subscription idle; only explicit DELETE or reaper expiry removes it.
    # ------------------------------------------------------------------

    async def _providers_for(self, sub_id, timeout=8.0):
        deadline = time.monotonic() + timeout
        providers = []
        while time.monotonic() < deadline:
            providers = self.request_json("GET", f"/ngsi-ld/v1/subscriptions/{quote(sub_id, safe='')}/providers")
            if providers:
                return providers
            await asyncio.sleep(0.25)
        return providers

    def test_34_one_shot_first_connection_creates_subscription_and_providers(self):
        asyncio.run(self._one_shot_first_connection_creates_subscription_and_providers())

    async def _one_shot_first_connection_creates_subscription_and_providers(self):
        entity_type = self.unique("ComdexOneShotFirst")
        entity_id = f"urn:ngsi-ld:{entity_type}:001"
        self.addCleanup(self.delete_entity_best_effort, entity_id, BROKER2_PORT)

        uri = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/ws"
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({
                "id": f"urn:subscription:{entity_type}", "type": "Subscription",
                "entities": [{"type": entity_type}], "@context": CONTEXT,
                "broker": BROKER_HOST, "port": BROKER2_PORT, "qos": QOS,
            }))
            _, subscribed = await self.recv_json(ws, timeout=5)
            self.assertEqual("subscribed", subscribed["status"])
            sub_id = subscribed["id"]
            self.addCleanup(self.delete_subscription_best_effort, sub_id)

            self.post_entity(self.entity_payload(entity_type, entity_id), BROKER2_PORT)
            _, msg = await self.wait_for_entity_message(ws, entity_id, timeout=8)
            self.assertEqual(entity_id, msg["id"])

            providers = await self._providers_for(sub_id)
            self.assertTrue(providers, "Provider child process must be created on first connection")

        subs = self.request_json("GET", "/ngsi-ld/v1/subscriptions")
        self.assertIn(sub_id, {s["id"] for s in subs})

    def test_35_one_shot_temporary_disconnect_keeps_subscription_and_providers_alive(self):
        asyncio.run(self._one_shot_temporary_disconnect_keeps_subscription_and_providers_alive())

    async def _one_shot_temporary_disconnect_keeps_subscription_and_providers_alive(self):
        entity_type = self.unique("ComdexOneShotIdle")
        entity_id = f"urn:ngsi-ld:{entity_type}:001"
        self.addCleanup(self.delete_entity_best_effort, entity_id, BROKER2_PORT)

        uri = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/ws"
        sub_id = None
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({
                "id": f"urn:subscription:{entity_type}", "type": "Subscription",
                "entities": [{"type": entity_type}], "@context": CONTEXT,
                "broker": BROKER_HOST, "port": BROKER2_PORT, "qos": QOS,
            }))
            _, subscribed = await self.recv_json(ws, timeout=5)
            sub_id = subscribed["id"]
            self.addCleanup(self.delete_subscription_best_effort, sub_id)
            self.post_entity(self.entity_payload(entity_type, entity_id), BROKER2_PORT)
            await self.wait_for_entity_message(ws, entity_id, timeout=8)
            providers_before = await self._providers_for(sub_id)
            self.assertTrue(providers_before)
        # WS closed here — must be treated as idle, not deleted.

        await asyncio.sleep(1.0)
        subs = self.request_json("GET", "/ngsi-ld/v1/subscriptions")
        self.assertIn(sub_id, {s["id"] for s in subs},
                     "Temporary WS disconnect must not delete the subscription (stop_subscription must not run)")
        providers_after = self.request_json("GET", f"/ngsi-ld/v1/subscriptions/{quote(sub_id, safe='')}/providers")
        self.assertTrue(providers_after, "Provider child processes must still be registered after disconnect")
        self.assertTrue(all(p["alive_process_count"] > 0 for p in providers_after),
                        "Provider child processes must still be alive after a temporary disconnect")

    def test_36_one_shot_reconnect_resumes_same_subscription_no_duplicates(self):
        asyncio.run(self._one_shot_reconnect_resumes_same_subscription_no_duplicates())

    async def _one_shot_reconnect_resumes_same_subscription_no_duplicates(self):
        entity_type = self.unique("ComdexOneShotReconnect")
        entity_id_1 = f"urn:ngsi-ld:{entity_type}:001"
        entity_id_2 = f"urn:ngsi-ld:{entity_type}:002"
        self.addCleanup(self.delete_entity_best_effort, entity_id_1, BROKER2_PORT)
        self.addCleanup(self.delete_entity_best_effort, entity_id_2, BROKER2_PORT)

        uri = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/ws"
        request = {
            "id": f"urn:subscription:{entity_type}", "type": "Subscription",
            "entities": [{"type": entity_type}], "@context": CONTEXT,
            "broker": BROKER_HOST, "port": BROKER2_PORT, "qos": QOS,
        }

        ws1 = await websockets.connect(uri)
        await ws1.send(json.dumps(request))
        _, subscribed = await self.recv_json(ws1, timeout=5)
        sub_id = subscribed["id"]
        self.addCleanup(self.delete_subscription_best_effort, sub_id)
        self.post_entity(self.entity_payload(entity_type, entity_id_1), BROKER2_PORT)
        await self.wait_for_entity_message(ws1, entity_id_1, timeout=8)
        providers_before = await self._providers_for(sub_id)
        pids_before = sorted(sum((p["process_ids"] for p in providers_before), []))
        self.assertTrue(pids_before)
        await ws1.close()
        await asyncio.sleep(0.5)

        async with websockets.connect(uri) as ws2:
            await ws2.send(json.dumps(request))
            _, subscribed2 = await self.recv_json(ws2, timeout=5)
            self.assertEqual("subscribed", subscribed2["status"])
            self.assertEqual(sub_id, subscribed2["id"], "Reconnect must resume the same subscription id")

            providers_after = self.request_json("GET", f"/ngsi-ld/v1/subscriptions/{quote(sub_id, safe='')}/providers")
            pids_after = sorted(sum((p["process_ids"] for p in providers_after), []))
            self.assertEqual(pids_before, pids_after, "Reconnect must reuse the existing provider child processes, not spawn duplicates")

            self.post_entity(self.entity_payload(entity_type, entity_id_2), BROKER2_PORT)
            _, msg = await self.wait_for_entity_message(ws2, entity_id_2, timeout=8)
            self.assertEqual(entity_id_2, msg["id"], "Notification stream must resume after reconnect")

    def test_37_one_shot_data_during_disconnect_remains_available_on_reconnect(self):
        asyncio.run(self._one_shot_data_during_disconnect_remains_available_on_reconnect())

    async def _one_shot_data_during_disconnect_remains_available_on_reconnect(self):
        entity_type = self.unique("ComdexOneShotQueued")
        entity_id_1 = f"urn:ngsi-ld:{entity_type}:001"
        entity_id_2 = f"urn:ngsi-ld:{entity_type}:002"
        self.addCleanup(self.delete_entity_best_effort, entity_id_1, BROKER2_PORT)
        self.addCleanup(self.delete_entity_best_effort, entity_id_2, BROKER2_PORT)

        uri = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/ws"
        request = {
            "id": f"urn:subscription:{entity_type}", "type": "Subscription",
            "entities": [{"type": entity_type}], "@context": CONTEXT,
            "broker": BROKER_HOST, "port": BROKER2_PORT, "qos": QOS,
        }

        ws1 = await websockets.connect(uri)
        await ws1.send(json.dumps(request))
        _, subscribed = await self.recv_json(ws1, timeout=5)
        sub_id = subscribed["id"]
        self.addCleanup(self.delete_subscription_best_effort, sub_id)
        self.post_entity(self.entity_payload(entity_type, entity_id_1), BROKER2_PORT)
        await self.wait_for_entity_message(ws1, entity_id_1, timeout=8)
        await self._providers_for(sub_id)
        await ws1.close()

        # Publish while nobody is connected — the provider chain and queue
        # must still be alive to capture this.
        self.post_entity(self.entity_payload(entity_type, entity_id_2), BROKER2_PORT)
        await asyncio.sleep(1.0)

        async with websockets.connect(uri) as ws2:
            await ws2.send(json.dumps(request))
            _, subscribed2 = await self.recv_json(ws2, timeout=5)
            self.assertEqual(sub_id, subscribed2["id"])
            _, msg = await self.wait_for_entity_message(ws2, entity_id_2, timeout=8)
            self.assertEqual(entity_id_2, msg["id"],
                             "Data published while disconnected must still reach the reconnected client")

    def test_38_one_shot_grace_expiry_reaps_subscription_and_providers(self):
        asyncio.run(self._one_shot_grace_expiry_reaps_subscription_and_providers())

    async def _one_shot_grace_expiry_reaps_subscription_and_providers(self):
        grace_seconds = 3
        port = 8012
        env = dict(os.environ)
        env["COMDEX_SUBSCRIPTION_IDLE_GRACE_SECONDS"] = str(grace_seconds)
        env["COMDEX_SUBSCRIPTION_REAPER_POLL_SECONDS"] = "1"
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "actionhandlerAPI:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.addCleanup(self._terminate_process, proc)
        base_url = f"http://127.0.0.1:{port}"
        ws_base = base_url.replace("http://", "ws://")
        self.assertTrue(wait_for_api(timeout=15.0, base_url=base_url))

        entity_type = self.unique("ComdexOneShotGraceExpiry")
        uri = f"{ws_base}/ngsi-ld/v1/subscriptions/ws"
        ws = await websockets.connect(uri)
        await ws.send(json.dumps({
            "id": f"urn:subscription:{entity_type}", "type": "Subscription",
            "entities": [{"type": entity_type}], "@context": CONTEXT,
            "broker": BROKER_HOST, "port": BROKER2_PORT, "qos": QOS,
        }))
        ack = json.loads(await asyncio.wait_for(ws.recv(), 10))
        sub_id = ack["id"]
        self.addCleanup(self.delete_subscription_best_effort, sub_id)
        await ws.close()

        deadline = time.monotonic() + grace_seconds * 4
        gone = False
        while time.monotonic() < deadline:
            r = requests.get(f"{base_url}/ngsi-ld/v1/subscriptions/{sub_id}", timeout=5)
            if r.status_code == 404:
                gone = True
                break
            await asyncio.sleep(0.3)
        self.assertTrue(gone, "Reaper must reclaim a one-shot subscription nobody reconnects to within the grace period")

    def test_39_one_shot_explicit_delete_removes_subscription_and_children(self):
        asyncio.run(self._one_shot_explicit_delete_removes_subscription_and_children())

    async def _one_shot_explicit_delete_removes_subscription_and_children(self):
        entity_type = self.unique("ComdexOneShotExplicitDelete")
        entity_id = f"urn:ngsi-ld:{entity_type}:001"
        self.addCleanup(self.delete_entity_best_effort, entity_id, BROKER2_PORT)

        uri = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/ws"
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({
                "id": f"urn:subscription:{entity_type}", "type": "Subscription",
                "entities": [{"type": entity_type}], "@context": CONTEXT,
                "broker": BROKER_HOST, "port": BROKER2_PORT, "qos": QOS,
            }))
            _, subscribed = await self.recv_json(ws, timeout=5)
            sub_id = subscribed["id"]
            self.post_entity(self.entity_payload(entity_type, entity_id), BROKER2_PORT)
            await self.wait_for_entity_message(ws, entity_id, timeout=8)
            await self._providers_for(sub_id)

        self.request_json("DELETE", f"/ngsi-ld/v1/subscriptions/{quote(sub_id, safe='')}")
        self.request_json("GET", f"/ngsi-ld/v1/subscriptions/{quote(sub_id, safe='')}", expected_status=404)
        self.request_json("GET", f"/ngsi-ld/v1/subscriptions/{quote(sub_id, safe='')}/providers", expected_status=404)

    def test_40_one_shot_duplicate_active_connection_rejected(self):
        asyncio.run(self._one_shot_duplicate_active_connection_rejected())

    async def _one_shot_duplicate_active_connection_rejected(self):
        entity_type = self.unique("ComdexOneShotDuplicate")
        entity_id_1 = f"urn:ngsi-ld:{entity_type}:001"
        entity_id_2 = f"urn:ngsi-ld:{entity_type}:002"
        self.addCleanup(self.delete_entity_best_effort, entity_id_1, BROKER2_PORT)
        self.addCleanup(self.delete_entity_best_effort, entity_id_2, BROKER2_PORT)

        uri = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/ws"
        request = {
            "id": f"urn:subscription:{entity_type}", "type": "Subscription",
            "entities": [{"type": entity_type}], "@context": CONTEXT,
            "broker": BROKER_HOST, "port": BROKER2_PORT, "qos": QOS,
        }

        async with websockets.connect(uri) as ws1:
            await ws1.send(json.dumps(request))
            _, subscribed = await self.recv_json(ws1, timeout=5)
            sub_id = subscribed["id"]
            self.addCleanup(self.delete_subscription_best_effort, sub_id)

            async with websockets.connect(uri) as ws2:
                await ws2.send(json.dumps(request))
                rejected = False
                try:
                    _, msg = await self.recv_json(ws2, timeout=5)
                    if "error" in msg:
                        rejected = True
                except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError):
                    rejected = True
                self.assertTrue(rejected, "A second one-shot WS on an already-connected subscription must be rejected")

            # Original connection must still work after the rejected second attempt.
            self.post_entity(self.entity_payload(entity_type, entity_id_1), BROKER2_PORT)
            _, msg = await self.wait_for_entity_message(ws1, entity_id_1, timeout=8)
            self.assertEqual(entity_id_1, msg["id"], "Original connection must be unaffected by the rejected duplicate")

    def test_41_one_shot_id_collision_with_different_params_rejected(self):
        asyncio.run(self._one_shot_id_collision_with_different_params_rejected())

    async def _one_shot_id_collision_with_different_params_rejected(self):
        entity_type_a = self.unique("ComdexOneShotCollideA")
        entity_type_b = self.unique("ComdexOneShotCollideB")
        entity_id = f"urn:ngsi-ld:{entity_type_a}:001"
        self.addCleanup(self.delete_entity_best_effort, entity_id, BROKER2_PORT)
        sub_id = f"urn:subscription:{self.unique('ComdexOneShotCollide')}"
        self.addCleanup(self.delete_subscription_best_effort, sub_id)

        uri = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/ws"
        ws1 = await websockets.connect(uri)
        await ws1.send(json.dumps({
            "id": sub_id, "type": "Subscription",
            "entities": [{"type": entity_type_a}], "@context": CONTEXT,
            "broker": BROKER_HOST, "port": BROKER2_PORT, "qos": QOS,
        }))
        _, subscribed = await self.recv_json(ws1, timeout=5)
        self.assertEqual(sub_id, subscribed["id"])
        await ws1.close()
        await asyncio.sleep(0.5)

        async with websockets.connect(uri) as ws2:
            await ws2.send(json.dumps({
                "id": sub_id, "type": "Subscription",
                "entities": [{"type": entity_type_b}], "@context": CONTEXT,
                "broker": BROKER_HOST, "port": BROKER2_PORT, "qos": QOS,
            }))
            rejected = False
            try:
                _, msg = await self.recv_json(ws2, timeout=5)
                if "error" in msg:
                    rejected = True
            except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError):
                rejected = True
            self.assertTrue(rejected, "Reconnect with different filter params for an existing id must be rejected")

        fetched = self.request_json("GET", f"/ngsi-ld/v1/subscriptions/{quote(sub_id, safe='')}")
        self.assertEqual(entity_type_a, fetched["type"], "Existing subscription must remain unchanged after a rejected id collision")

    # ------------------------------------------------------------------
    # Malformed-payload resilience (test_42+).
    #
    # Regression for: _parse_payload() let json.JSONDecodeError/ValueError/
    # SyntaxError/UnicodeDecodeError escape, and subscribe().on_message()
    # called recreate_single_entity() unprotected - one malformed retained
    # MQTT message (bad JSON/literal, invalid UTF-8, or an unexpected
    # nested shape) could raise out of the MQTT callback and crash/kill the
    # provider subscription's child process. See actionhandler.py
    # PayloadParseError, _parse_payload(), recreate_single_entity() and
    # subscribe().on_message() for the fix.
    # ------------------------------------------------------------------

    def publish_raw(self, broker, port, topic, payload, retain=True, qos=1):
        client = mqtt.Client()
        client.connect(broker, port)
        client.loop_start()
        info = client.publish(topic, payload, qos=qos, retain=retain)
        info.wait_for_publish(10)
        client.loop_stop()
        client.disconnect()

    def attribute_topic(self, area, context, entity_type, entity_id, attr):
        hlink = context.replace("/", "§")
        return f"{area}/entities/{hlink}/{entity_type}/LNA/{entity_id}/{attr}"

    async def _wait_for_provider_pid(self, subscription_id, timeout=8.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            providers = self.request_json("GET", f"/ngsi-ld/v1/subscriptions/{quote(subscription_id, safe='')}/providers")
            pids = sorted(pid for p in providers for pid in p.get("process_ids", []) if pid is not None)
            if pids:
                return pids
            await asyncio.sleep(0.25)
        return []

    def test_42_malformed_payload_skipped_valid_messages_before_and_after_survive(self):
        asyncio.run(self._malformed_payload_skipped_valid_messages_before_and_after_survive())

    async def _malformed_payload_skipped_valid_messages_before_and_after_survive(self):
        """The most important regression: BAD then VALID must not affect the
        provider subscription's child process - same PID, no reconnect."""
        entity_type = self.unique("ComdexMalformed")
        entity_id = f"urn:ngsi-ld:{entity_type}:001"
        subscription_id = f"urn:subscription:{entity_type}"
        self.addCleanup(self.delete_entity_best_effort, entity_id, BROKER2_PORT)
        self.addCleanup(self.delete_subscription_best_effort, subscription_id)

        self.request_json(
            "POST",
            f"/ngsi-ld/v1/subscriptions?broker={BROKER_HOST}&port={BROKER2_PORT}&qos={QOS}&my_area={AREA}",
            expected_status=201,
            json={"id": subscription_id, "type": "Subscription", "entities": [{"type": entity_type}], "@context": CONTEXT},
        )

        uri = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/{quote(subscription_id, safe='')}/ws"
        async with websockets.connect(uri) as ws:
            await self.recv_json(ws, timeout=3)

            # Valid entity #1: establishes the provider child process.
            self.post_entity(self.entity_payload(entity_type, entity_id), BROKER2_PORT)
            _, msg = await self.wait_for_entity_message(ws, entity_id, timeout=8)
            self.assertEqual(entity_id, msg["id"])

            pids_before = await self._wait_for_provider_pid(subscription_id)
            self.assertTrue(pids_before, "No provider child process was registered before the malformed publish")

            # BAD: neither valid JSON nor a valid Python literal.
            self.publish_raw(BROKER_HOST, BROKER2_PORT,
                             self.attribute_topic(AREA, PRIMARY_CONTEXT, entity_type, entity_id, "badAttr"),
                             b'{this is neither json nor a python literal ]]')
            await asyncio.sleep(1.5)

            providers_after_bad = self.request_json("GET", f"/ngsi-ld/v1/subscriptions/{quote(subscription_id, safe='')}/providers")
            pids_after_bad = sorted(pid for p in providers_after_bad for pid in p.get("process_ids", []) if pid is not None)
            self.assertEqual(pids_before, pids_after_bad,
                             "Provider child process must survive a malformed payload with the SAME PID - no crash, no restart")
            self.assertTrue(all(p["alive_process_count"] > 0 for p in providers_after_bad),
                            "Provider child process must still be alive after the malformed payload")

            # VALID: a second entity must still be delivered normally on the
            # same connection, through the same still-alive child process.
            entity_id_2 = f"urn:ngsi-ld:{entity_type}:002"
            self.addCleanup(self.delete_entity_best_effort, entity_id_2, BROKER2_PORT)
            self.post_entity(self.entity_payload(entity_type, entity_id_2), BROKER2_PORT)
            _, msg2 = await self.wait_for_entity_message(ws, entity_id_2, timeout=8)
            self.assertEqual(entity_id_2, msg2["id"], "A valid message published after a malformed one must still be delivered")

            pids_final = self.request_json("GET", f"/ngsi-ld/v1/subscriptions/{quote(subscription_id, safe='')}/providers")
            pids_final = sorted(pid for p in pids_final for pid in p.get("process_ids", []) if pid is not None)
            self.assertEqual(pids_before, pids_final, "PID must still be identical after processing the valid follow-up message")

    def test_43_malformed_payload_on_one_provider_does_not_affect_another(self):
        asyncio.run(self._malformed_payload_on_one_provider_does_not_affect_another())

    async def _malformed_payload_on_one_provider_does_not_affect_another(self):
        type_a = self.unique("ComdexMalformedA")
        type_b = self.unique("ComdexMalformedB")
        id_a = f"urn:ngsi-ld:{type_a}:001"
        id_b = f"urn:ngsi-ld:{type_b}:001"
        sub_a = f"urn:subscription:{type_a}"
        sub_b = f"urn:subscription:{type_b}"
        self.addCleanup(self.delete_entity_best_effort, id_a, BROKER1_PORT)
        self.addCleanup(self.delete_entity_best_effort, id_b, BROKER2_PORT)
        self.addCleanup(self.delete_subscription_best_effort, sub_a)
        self.addCleanup(self.delete_subscription_best_effort, sub_b)

        for sub_id, port in ((sub_a, BROKER1_PORT), (sub_b, BROKER2_PORT)):
            self.request_json(
                "POST",
                f"/ngsi-ld/v1/subscriptions?broker={BROKER_HOST}&port={port}&qos={QOS}&my_area={AREA}",
                expected_status=201,
                json={"id": sub_id, "type": "Subscription",
                     "entities": [{"type": type_a if sub_id == sub_a else type_b}], "@context": CONTEXT},
            )

        uri_a = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/{quote(sub_a, safe='')}/ws"
        uri_b = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/{quote(sub_b, safe='')}/ws"
        async with websockets.connect(uri_a) as ws_a, websockets.connect(uri_b) as ws_b:
            await self.recv_json(ws_a, timeout=3)
            await self.recv_json(ws_b, timeout=3)

            self.post_entity(self.entity_payload(type_a, id_a), BROKER1_PORT)
            await self.wait_for_entity_message(ws_a, id_a, timeout=8)
            self.post_entity(self.entity_payload(type_b, id_b), BROKER2_PORT)
            await self.wait_for_entity_message(ws_b, id_b, timeout=8)

            pids_b_before = await self._wait_for_provider_pid(sub_b)
            self.assertTrue(pids_b_before)

            # Malformed publish ONLY on provider A's broker/topic.
            self.publish_raw(BROKER_HOST, BROKER1_PORT,
                             self.attribute_topic(AREA, PRIMARY_CONTEXT, type_a, id_a, "badAttr"),
                             b'not json { not python either [[[')
            await asyncio.sleep(1.5)

            pids_b_after = self.request_json("GET", f"/ngsi-ld/v1/subscriptions/{quote(sub_b, safe='')}/providers")
            pids_b_after = sorted(pid for p in pids_b_after for pid in p.get("process_ids", []) if pid is not None)
            self.assertEqual(pids_b_before, pids_b_after, "Provider B's child process must be unaffected by provider A's malformed message")

            id_b_2 = f"urn:ngsi-ld:{type_b}:002"
            self.addCleanup(self.delete_entity_best_effort, id_b_2, BROKER2_PORT)
            self.post_entity(self.entity_payload(type_b, id_b_2), BROKER2_PORT)
            _, msg = await self.wait_for_entity_message(ws_b, id_b_2, timeout=8)
            self.assertEqual(id_b_2, msg["id"], "Provider B must keep delivering data unaffected by provider A's malformed message")

    def test_44_invalid_utf8_bytes_rejected_worker_survives(self):
        asyncio.run(self._invalid_utf8_bytes_rejected_worker_survives())

    async def _invalid_utf8_bytes_rejected_worker_survives(self):
        entity_type = self.unique("ComdexBadUtf8")
        entity_id = f"urn:ngsi-ld:{entity_type}:001"
        subscription_id = f"urn:subscription:{entity_type}"
        self.addCleanup(self.delete_entity_best_effort, entity_id, BROKER2_PORT)
        self.addCleanup(self.delete_subscription_best_effort, subscription_id)

        self.request_json(
            "POST",
            f"/ngsi-ld/v1/subscriptions?broker={BROKER_HOST}&port={BROKER2_PORT}&qos={QOS}&my_area={AREA}",
            expected_status=201,
            json={"id": subscription_id, "type": "Subscription", "entities": [{"type": entity_type}], "@context": CONTEXT},
        )

        uri = f"{WS_BASE_URL}/ngsi-ld/v1/subscriptions/{quote(subscription_id, safe='')}/ws"
        async with websockets.connect(uri) as ws:
            await self.recv_json(ws, timeout=3)
            self.post_entity(self.entity_payload(entity_type, entity_id), BROKER2_PORT)
            await self.wait_for_entity_message(ws, entity_id, timeout=8)
            pids_before = await self._wait_for_provider_pid(subscription_id)
            self.assertTrue(pids_before)

            # Invalid UTF-8 byte sequence - must be rejected during decode,
            # not crash the callback.
            self.publish_raw(BROKER_HOST, BROKER2_PORT,
                             self.attribute_topic(AREA, PRIMARY_CONTEXT, entity_type, entity_id, "badUtf8Attr"),
                             b'\xff\xfe\x00\x01garbage-not-utf8')
            await asyncio.sleep(1.5)

            providers = self.request_json("GET", f"/ngsi-ld/v1/subscriptions/{quote(subscription_id, safe='')}/providers")
            pids_after = sorted(pid for p in providers for pid in p.get("process_ids", []) if pid is not None)
            self.assertEqual(pids_before, pids_after, "Invalid UTF-8 payload must not kill the provider child process")

            entity_id_2 = f"urn:ngsi-ld:{entity_type}:002"
            self.addCleanup(self.delete_entity_best_effort, entity_id_2, BROKER2_PORT)
            self.post_entity(self.entity_payload(entity_type, entity_id_2), BROKER2_PORT)
            _, msg = await self.wait_for_entity_message(ws, entity_id_2, timeout=8)
            self.assertEqual(entity_id_2, msg["id"], "Valid message after invalid UTF-8 payload must still be delivered")


if __name__ == "__main__":
    unittest.main(verbosity=2)