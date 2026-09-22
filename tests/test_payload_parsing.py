"""Unit regression tests for the payload-parsing/entity-reconstruction
hardening in actionhandler.py: a malformed individual MQTT message must
never raise out of _parse_payload()/recreate_single_entity() in a way that
would escape an on_message() callback and kill the subscription worker.

No broker/API server needed - these exercise the pure functions directly.
See tests/test_comdex_api.py (test_42+) for the real-broker end-to-end
regression proving a provider child process survives a bad payload.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import actionhandler as ah


class FakeMsg:
    def __init__(self, topic, payload):
        self.topic = topic
        self.payload = payload


CONTEXT = "https:§§uri.etsi.org§ngsi-ld§v1§ngsi-ld-core-context.jsonld".replace("§", "/")
AREA = "unknown_area"
TYPE = "GtfsAgency"
ENTITY_ID = "urn:ngsi-ld:GtfsAgency:001"
HLINK = "https§§uri.etsi.org§ngsi-ld§v1§ngsi-ld-core-context.jsonld"


def attr_topic(attr):
    return f"{AREA}/entities/{HLINK}/{TYPE}/LNA/{ENTITY_ID}/{attr}"


class ParsePayloadTests(unittest.TestCase):
    def test_a_valid_json_parsed_normally(self):
        self.assertEqual({"a": 1, "b": [1, 2]}, ah._parse_payload(b'{"a": 1, "b": [1, 2]}'))

    def test_b_legacy_python_literal_still_accepted(self):
        # Not valid JSON (single quotes), but a valid Python literal - this
        # is the pre-existing fallback format and must keep working.
        self.assertEqual({'a': 1, 'b': (1, 2)}, ah._parse_payload(b"{'a': 1, 'b': (1, 2)}"))

    def test_c_malformed_json_and_literal_raises_payload_parse_error(self):
        with self.assertRaises(ah.PayloadParseError):
            ah._parse_payload(b'{not valid json or python literal')

    def test_d_invalid_utf8_bytes_raise_payload_parse_error(self):
        with self.assertRaises(ah.PayloadParseError):
            ah._parse_payload(b'\xff\xfe\x00\x01not-utf8')

    def test_payload_parse_error_preview_is_bounded_not_full_payload(self):
        huge = b'{' + b'x' * 10000
        try:
            ah._parse_payload(huge)
            self.fail("expected PayloadParseError")
        except ah.PayloadParseError as exc:
            self.assertLess(len(exc.preview), 1000, "preview must not include the full huge payload")

    def test_diagnostics_counters_exist_and_are_incrementable(self):
        before = ah.diagnostics.get('payload_parse_errors', 0)
        ah._bump_diagnostic('payload_parse_errors')
        self.assertEqual(before + 1, ah.diagnostics['payload_parse_errors'])


class RecreateSingleEntityTests(unittest.TestCase):
    def test_valid_messages_reconstruct_entity(self):
        messages = [FakeMsg(attr_topic("agencyName"), b'{"type": "Property", "value": "Acme"}')]
        entity = ah.recreate_single_entity(messages)
        self.assertEqual(ENTITY_ID, entity['id'])
        self.assertEqual({"type": "Property", "value": "Acme"}, entity['agencyName'])

    def test_g_malformed_message_among_valid_ones_is_skipped_not_fatal(self):
        messages = [
            FakeMsg(attr_topic("agencyName"), b'{"type": "Property", "value": "Acme"}'),
            FakeMsg(attr_topic("language"), b'{not valid json or python'),
            FakeMsg(attr_topic("otherAttr"), b'{"type": "Property", "value": "ok"}'),
        ]
        entity = ah.recreate_single_entity(messages)
        self.assertIsNotNone(entity, "a malformed fragment must not abort the whole reconstruction")
        self.assertEqual({"type": "Property", "value": "Acme"}, entity['agencyName'])
        self.assertEqual({"type": "Property", "value": "ok"}, entity['otherAttr'])
        self.assertNotIn('language', entity, "the malformed fragment itself must be dropped, not invented")

    def test_g_malformed_nested_geo_value_shape_does_not_crash(self):
        # georel path does data2["value"]["type"]/["coordinates"] - a
        # payload that parses fine as JSON but isn't a geo object (wrong
        # shape) must not crash reconstruction.
        messages = [FakeMsg(attr_topic("location"), b'{"type": "Property", "value": "not-a-geo-object"}')]
        entity = ah.recreate_single_entity(messages, georel="intersects", geometry="Point",
                                           coordinates="[0, 0]", geoproperty="location")
        # No crash is the point of this test; a geo mismatch/skip legitimately
        # returns None or a dict without the malformed attribute.
        if entity is not None:
            self.assertNotIn('location', entity)

    def test_reconstruction_error_counter_increments_on_malformed_fragment(self):
        before = ah.diagnostics.get('reconstruction_errors', 0)
        messages = [FakeMsg(attr_topic("location"), b'{"type": "Property", "value": "not-a-geo-object"}')]
        ah.recreate_single_entity(messages, georel="intersects", geometry="Point",
                                  coordinates="[0, 0]", geoproperty="location")
        self.assertGreaterEqual(ah.diagnostics.get('reconstruction_errors', 0), before + 1)


if __name__ == "__main__":
    unittest.main()
