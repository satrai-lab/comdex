import asyncio
import json
import multiprocessing
import queue
import threading
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Path, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from actionhandler import (
    active_subscriptions,
    batch_create,
    batch_delete,
    batch_update,
    batch_upsert,
    close_publisher_clients,
    default_broker_adress,
    default_broker_port,
    delete_entity,
    delete_entity_attr,
    get_entities,
    patch_entity,
    patch_entity_attr,
    post_entity,
    post_entity_upsert,
    post_subscription,
    list_subscription_providers,
    stop_all_subscriptions,
    stop_subscription_provider,
    stop_subscription,
)

# Per-subscription notification queues (populated by POST /subscriptions)
notification_queues: dict = {}
QUEUE_WAIT_SECONDS = 0.005

app = FastAPI(
    title="ComDeX NGSI-LD API",
    description="NGSI-LD compliant API over MQTT using ComDeX",
    version="0.6.1",
)


@app.on_event("shutdown")
def shutdown_runtime_resources():
    stopped = stop_all_subscriptions()
    if stopped:
        print(f"Stopped subscriptions on shutdown: {stopped}")
    notification_queues.clear()
    close_publisher_clients()


def notification_payload_for_subscription(subscription_id: str, item):
    if not isinstance(item, dict) or not item.get("__comdex_notification__"):
        return item

    provider_key = item.get("provider_key")
    subscription = active_subscriptions.get(subscription_id)
    if subscription is not None and provider_key:
        provider_lock = subscription.get("provider_lock")
        if provider_lock is not None:
            with provider_lock:
                if provider_key in subscription.get("disabled_provider_keys", set()):
                    return None

    return item.get("payload")


async def send_next_notifications(websocket: WebSocket, notification_q, loop, subscription_id: str):
    while True:
        item = await loop.run_in_executor(
            None, lambda: notification_q.get(timeout=QUEUE_WAIT_SECONDS)
        )
        entity = notification_payload_for_subscription(subscription_id, item)
        if entity is None:
            continue
        await websocket.send_json(entity)
        break

    while True:
        try:
            item = notification_q.get_nowait()
        except queue.Empty:
            break
        entity = notification_payload_for_subscription(subscription_id, item)
        if entity is None:
            continue
        await websocket.send_json(entity)


def drain_notification_queue(subscription_id: str, quiet_seconds: float = 0.1, max_seconds: float = 1.0):
    notification_q = notification_queues.get(subscription_id)
    if notification_q is None:
        return 0
    drained = 0
    deadline = time.monotonic() + max_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return drained
        try:
            notification_q.get(timeout=min(quiet_seconds, remaining))
            drained += 1
        except queue.Empty:
            return drained


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

@app.post("/ngsi-ld/v1/entities", status_code=201, tags=["Entities"],
          summary="Create a new NGSI-LD entity")
def create_entity(
    body: dict,
    broker: str = Query(default_broker_adress, description="MQTT broker address"),
    port: int = Query(default_broker_port, description="MQTT broker port"),
    qos: int = Query(0, description="MQTT QoS level (0, 1 or 2)"),
    my_area: str = Query("unknown_area"),
    my_loc: str = Query("unknown_location"),
):
    try:
        post_entity_upsert(body, my_area, broker, port, qos, my_loc)
        return {"status": "upserted", "id": body.get("id")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ngsi-ld/v1/entities", tags=["Entities"],
         summary="Query NGSI-LD entities")
def query_entities(
    type: Optional[str] = Query(None, description="Entity type filter (comma-separated)"),
    id: Optional[str] = Query(None, description="Entity ID filter"),
    attrs: Optional[str] = Query(None, description="Attributes to return (comma-separated)"),
    q: Optional[str] = Query(None, description="Attribute query (e.g. temperature>20)"),
    georel: Optional[str] = Query(None, description="Geo-relation (within, intersects, ...)"),
    geometry: Optional[str] = Query(None, description="Geometry type (Point, Polygon, ...)"),
    coordinates: Optional[str] = Query(None, description="Geometry coordinates (JSON array)"),
    geoproperty: str = Query("location", description="Entity property to apply geo-filter on"),
    area: Optional[str] = Query(None, description="ComDeX area filter (comma-separated)"),
    limit: int = Query(1800, description="Maximum number of entities to return"),
    broker: str = Query(default_broker_adress),
    port: int = Query(default_broker_port),
    hlink: str = Query("+", description="Context link (HLink)"),
):
    results = get_entities(
        broker=broker,
        port=port,
        hlink=hlink,
        entity_type=type.split(",") if type else None,
        entity_id=id,
        attrs=attrs.split(",") if attrs else None,
        q=q,
        georel=georel,
        geometry=geometry,
        coordinates=coordinates,
        geoproperty=geoproperty,
        area=area.split(",") if area else None,
        limit=limit,
    )
    return results


@app.delete("/ngsi-ld/v1/entities/{entityId}", tags=["Entities"],
            summary="Delete an NGSI-LD entity")
def remove_entity(
    entityId: str = Path(..., description="Entity ID"),
    broker: str = Query(default_broker_adress),
    port: int = Query(default_broker_port),
    hlink: str = Query("+"),
    my_area: str = Query("unknown_area"),
):
    try:
        delete_entity(entityId, broker, port, hlink, my_area)
        return {"status": "deleted", "id": entityId}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.patch("/ngsi-ld/v1/entities/{entityId}/attrs", tags=["Entities"],
           summary="Patch all attributes of an entity")
def update_entity_attrs(
    entityId: str = Path(...),
    body: dict = ...,
    broker: str = Query(default_broker_adress),
    port: int = Query(default_broker_port),
    hlink: str = Query("+"),
    qos: int = Query(0),
    my_area: str = Query("unknown_area"),
    validate_exists: bool = Query(True, description="Check entity existence before patching"),
    entity_type: Optional[str] = Query(None, description="Required when validate_exists=false"),
):
    try:
        patch_entity(entityId, body, broker, port, hlink, qos, my_area, validate_exists, entity_type)
        return {"status": "updated", "id": entityId}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.patch("/ngsi-ld/v1/entities/{entityId}/attrs/{attrName}", tags=["Entities"],
           summary="Patch a single attribute of an entity")
def update_entity_attr(
    entityId: str = Path(...),
    attrName: str = Path(...),
    body: dict = ...,
    broker: str = Query(default_broker_adress),
    port: int = Query(default_broker_port),
    hlink: str = Query("+"),
    qos: int = Query(0),
    my_area: str = Query("unknown_area"),
    validate_exists: bool = Query(True, description="Check entity existence before patching"),
    entity_type: Optional[str] = Query(None, description="Required when validate_exists=false"),
):
    try:
        patch_entity_attr(entityId, attrName, body, broker, port, hlink, qos, my_area, validate_exists, entity_type)
        return {"status": "updated", "id": entityId, "attr": attrName}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/ngsi-ld/v1/entities/{entityId}/attrs/{attrName}", tags=["Entities"],
            summary="Delete a single attribute of an entity")
def remove_entity_attr(
    entityId: str = Path(...),
    attrName: str = Path(...),
    broker: str = Query(default_broker_adress),
    port: int = Query(default_broker_port),
    hlink: str = Query("+"),
    my_area: str = Query("unknown_area"),
):
    delete_entity_attr(entityId, attrName, broker, port, hlink, my_area)
    return {"status": "deleted", "id": entityId, "attr": attrName}


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------

@app.post("/ngsi-ld/v1/entityOperations/create", tags=["Batch"],
          summary="Batch create entities")
def batch_create_entities(
    body: List[dict],
    broker: str = Query(default_broker_adress),
    port: int = Query(default_broker_port),
    qos: int = Query(0),
    my_area: str = Query("unknown_area"),
    my_loc: str = Query("unknown_location"),
):
    batch_create(body, broker, port, qos, my_area, my_loc)
    return {"status": "created", "count": len(body)}


@app.post("/ngsi-ld/v1/entityOperations/update", tags=["Batch"],
          summary="Batch update entities")
def batch_update_entities(
    body: List[dict],
    broker: str = Query(default_broker_adress),
    port: int = Query(default_broker_port),
    qos: int = Query(0),
    my_area: str = Query("unknown_area"),
    my_loc: str = Query("unknown_location"),
):
    batch_update(body, broker, port, qos, my_area, my_loc)
    return {"status": "updated", "count": len(body)}


@app.post("/ngsi-ld/v1/entityOperations/upsert", tags=["Batch"],
          summary="Batch upsert entities")
def batch_upsert_entities(
    body: List[dict],
    broker: str = Query(default_broker_adress),
    port: int = Query(default_broker_port),
    qos: int = Query(0),
    my_area: str = Query("unknown_area"),
    my_loc: str = Query("unknown_location"),
):
    batch_upsert(body, broker, port, qos, my_area, my_loc)
    return {"status": "upserted", "count": len(body)}


@app.post("/ngsi-ld/v1/entityOperations/delete", tags=["Batch"],
          summary="Batch delete entities by ID list")
def batch_delete_entities(
    body: List[str],
    broker: str = Query(default_broker_adress),
    port: int = Query(default_broker_port),
    hlink: str = Query("+"),
    my_area: str = Query("unknown_area"),
):
    batch_delete(body, broker, port, hlink, my_area)
    return {"status": "deleted", "count": len(body)}


# ---------------------------------------------------------------------------
# Subscriptions — REST
# ---------------------------------------------------------------------------

@app.post("/ngsi-ld/v1/subscriptions", status_code=201, tags=["Subscriptions"],
          summary="Create a subscription by type, id, or watched attributes")
def create_subscription(
    body: dict,
    broker: str = Query(default_broker_adress, description="MQTT broker to watch for advertisements"),
    port: int = Query(default_broker_port),
    qos: int = Query(0),
    my_area: str = Query("unknown_area"),
):
    """
    Body follows the NGSI-LD Subscription format:

        {
            "id": "urn:subscription:1",
            "type": "Subscription",
            "entities": [{"type": "GtfsAgency"}],
            "watchedAttributes": ["agencyName", "language"],
            "@context": "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld"
        }

    You can filter by:
    - **type** only → `"entities": [{"type": "SomeType"}]`
    - **id** only → `"entities": [{"id": "urn:ngsi-ld:..."}]`
    - **type + id** → `"entities": [{"type": "SomeType", "id": "urn:ngsi-ld:..."}]`
    - **watchedAttributes** → only notify on specific attribute changes

    After creation connect to `WS /ngsi-ld/v1/subscriptions/{id}/ws` to stream notifications.
    """
    try:
        notification_q = multiprocessing.Queue()
        sub_id = post_subscription(body, broker, port, qos, my_area=my_area,
                                   notification_queue=notification_q)
        notification_queues[sub_id] = notification_q
        return {"status": "created", "id": sub_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ngsi-ld/v1/subscriptions", tags=["Subscriptions"],
         summary="List all active subscriptions")
def list_subscriptions():
    return [
        {
            "id":         sub_id,
            "type":       info["type"],
            "entity_id":  info["entity_id"],
            "broker":     info["broker"],
            "port":       info["port"],
            "created_at": info["created_at"],
            "expires":    info["expires"],
            "status":     info["status"],
            "providers":   list_subscription_providers(sub_id) or [],
        }
        for sub_id, info in active_subscriptions.items()
    ]


@app.get("/ngsi-ld/v1/subscriptions/{subscriptionId}", tags=["Subscriptions"],
         summary="Get a single subscription by ID")
def get_subscription(subscriptionId: str = Path(...)):
    if subscriptionId not in active_subscriptions:
        raise HTTPException(status_code=404, detail=f"Subscription {subscriptionId} not found")
    info = active_subscriptions[subscriptionId]
    return {
        "id":         subscriptionId,
        "type":       info["type"],
        "entity_id":  info["entity_id"],
        "broker":     info["broker"],
        "port":       info["port"],
        "created_at": info["created_at"],
        "expires":    info["expires"],
        "status":     info["status"],
        "providers":   list_subscription_providers(subscriptionId) or [],
    }


@app.get("/ngsi-ld/v1/subscriptions/{subscriptionId}/providers", tags=["Subscriptions"],
         summary="List provider child processes for a subscription")
def get_subscription_providers(subscriptionId: str = Path(...)):
    providers = list_subscription_providers(subscriptionId)
    if providers is None:
        raise HTTPException(status_code=404, detail=f"Subscription {subscriptionId} not found")
    return providers


@app.delete("/ngsi-ld/v1/subscriptions/{subscriptionId}/providers", tags=["Subscriptions"],
            summary="Stop provider child process(es) for a subscription")
def remove_subscription_provider(
    subscriptionId: str = Path(...),
    provider_broker: Optional[str] = Query(None, description="Provider broker address to stop"),
    provider_port: Optional[int] = Query(None, description="Provider broker port to stop"),
    area: Optional[str] = Query(None, description="Provider area filter"),
    context: Optional[str] = Query(None, description="Provider context/HLink filter"),
    entity_type: Optional[str] = Query(None, description="Provider entity type filter"),
):
    if all(value is None for value in (provider_broker, provider_port, area, context, entity_type)):
        raise HTTPException(status_code=400, detail="At least one provider filter is required")
    stopped = stop_subscription_provider(
        subscriptionId,
        broker=provider_broker,
        port=provider_port,
        area=area,
        context=context,
        entity_type=entity_type,
    )
    if stopped is None:
        raise HTTPException(status_code=404, detail=f"Subscription {subscriptionId} not found")
    if not stopped:
        raise HTTPException(status_code=404, detail="No matching provider child process found")
    drained = drain_notification_queue(subscriptionId)
    remaining = list_subscription_providers(subscriptionId) or []
    return {
        "status": "provider_stopped",
        "id": subscriptionId,
        "providers": stopped,
        "remaining_providers": remaining,
        "drained_notifications": drained,
    }


@app.delete("/ngsi-ld/v1/subscriptions/{subscriptionId}", tags=["Subscriptions"],
            summary="Stop and remove a subscription")
def remove_subscription(subscriptionId: str = Path(...)):
    if not stop_subscription(subscriptionId):
        raise HTTPException(status_code=404, detail=f"Subscription {subscriptionId} not found")
    notification_queues.pop(subscriptionId, None)
    return {"status": "stopped", "id": subscriptionId}


# ---------------------------------------------------------------------------
# Subscriptions — WebSocket: stream notifications for an existing subscription
# ---------------------------------------------------------------------------

@app.websocket("/ngsi-ld/v1/subscriptions/{subscriptionId}/ws")
async def subscription_notifications_ws(websocket: WebSocket, subscriptionId: str):
    """
    Stream live notifications for a subscription created via POST /subscriptions.

    Flow:
      1. POST /ngsi-ld/v1/subscriptions  → get subscription id
      2. Connect here with that id
      3. Receive entity JSON objects as they arrive
      4. Disconnect anytime — subscription keeps running in background
    """
    if subscriptionId not in active_subscriptions:
        await websocket.close(code=1008)
        return

    notification_q = notification_queues.get(subscriptionId)
    if notification_q is None:
        await websocket.close(code=1011)
        return

    await websocket.accept()
    await websocket.send_json({"status": "connected", "id": subscriptionId})

    info = active_subscriptions[subscriptionId]
    entity_type = info.get("type")
    entity_id = info.get("entity_id")
    watched_attributes = info.get("watched_attributes")
    snapshot = get_entities(
        broker=info["broker"],
        port=info["port"],
        hlink="+",
        entity_type=[entity_type] if entity_type and entity_type != "+" else None,
        entity_id=entity_id or None,
        attrs=watched_attributes,
        limit=1800,
        fast=True,
    )
    for entity in snapshot:
        await websocket.send_json(entity)

    loop = asyncio.get_event_loop()
    try:
        while subscriptionId in active_subscriptions:
            try:
                await send_next_notifications(websocket, notification_q, loop, subscriptionId)
            except queue.Empty:
                continue
        await websocket.send_json({"status": "stopped", "id": subscriptionId})
        await websocket.close(code=1000)
    except WebSocketDisconnect:
        pass  # subscription keeps running — client can reconnect


# ---------------------------------------------------------------------------
# Subscriptions — WebSocket: create + stream in one connection
# ---------------------------------------------------------------------------

@app.websocket("/ngsi-ld/v1/subscriptions/ws")
async def subscription_websocket(websocket: WebSocket):
    """
    Create a subscription and receive its notifications in a single WebSocket connection.

    Flow:
      1. Connect.
      2. Send subscription JSON (NGSI-LD format + optional "broker"/"port"/"qos" fields).
      3. Receive: {"status": "subscribed", "id": "<sub_id>"}
      4. Receive entity JSON objects as they arrive.
      5. Disconnect → subscription is automatically stopped.
    """
    await websocket.accept()

    try:
        raw = await websocket.receive_json()
    except Exception:
        await websocket.close(code=1003)
        return

    broker = raw.get("broker", default_broker_adress)
    port   = raw.get("port",   default_broker_port)
    qos    = raw.get("qos",    0)

    notification_q = multiprocessing.Queue()

    try:
        sub_id = post_subscription(raw, broker, port, qos, notification_queue=notification_q)
    except ValueError as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close(code=1008)
        return

    await websocket.send_json({"status": "subscribed", "id": sub_id})

    loop = asyncio.get_event_loop()
    try:
        while True:
            try:
                await send_next_notifications(websocket, notification_q, loop, sub_id)
            except queue.Empty:
                continue
    except WebSocketDisconnect:
        stop_subscription(sub_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
