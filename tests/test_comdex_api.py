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


def api_is_ready():
    try:
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=1)
        return response.status_code == 200
    except requests.RequestException:
        return False


def wait_for_api(timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if api_is_ready():
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
        """WS /subscriptions/ws creates a subscription inline and auto-stops it on disconnect."""
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
        # The server's send loop only discovers the disconnect when it next attempts
        # a send on the closed socket.  Posting a second entity forces that attempt.

        entity_id2 = f"urn:ngsi-ld:{entity_type}:002"
        self.addCleanup(self.delete_entity_best_effort, entity_id2, BROKER2_PORT)
        self.post_entity(self.entity_payload(entity_type, entity_id2), BROKER2_PORT)
        await asyncio.sleep(1.5)  # wait for server to detect disconnect and stop subscription

        subs = self.request_json("GET", "/ngsi-ld/v1/subscriptions")
        active_ids = {s["id"] for s in subs}
        self.assertNotIn(sub_id, active_ids, "One-shot subscription must be removed after WS disconnect")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
