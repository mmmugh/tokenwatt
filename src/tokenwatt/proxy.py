from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
from typing import Callable

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from tokenwatt.meter import EnergyMeter
from tokenwatt.idle import IdleBaseline
from tokenwatt.rate import FlatRate
from tokenwatt.coldstart import ColdStartDetector
from tokenwatt.usage import usage_from_response_json, SelfCounter, TokenUsage
from tokenwatt.ledger import Ledger, LedgerRow
from tokenwatt.router import Router
from tokenwatt.reqtype import classify_request
from tokenwatt.log import event

logger = logging.getLogger("tokenwatt.proxy")

# Hop-by-hop headers that must not be forwarded.
_DROP = {"content-length", "transfer-encoding", "connection", "host"}


def create_app(*, router: Router, meter: EnergyMeter, idle: IdleBaseline,
               ledger: Ledger, rate: FlatRate, client: httpx.AsyncClient,
               detector: ColdStartDetector,
               serialize_lock=None,
               _label_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
               lifespan=None) -> Starlette:

    async def forward(request: Request) -> Response:
        path = request.path_params["path"]
        raw = await request.body()
        try:
            req_json = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            req_json = {}
        model = req_json.get("model", "unknown")
        is_stream = bool(req_json.get("stream"))
        req_type = classify_request(path, req_json)

        route = router.resolve(model)
        if route is None:
            logger.warning("req.no_route", extra=event(model=model, routes=router.route_names()))
            return JSONResponse(
                {"error": {
                    "message": f"no route for model {model!r}; configured routes: {router.route_names()}",
                    "type": "no_route",
                }},
                status_code=404,
            )
        # Record the ACTUAL requested model (what users compare), not the route alias, so a
        # multi-model backend (e.g. LM Studio on one port) yields one row per model. Fall back
        # to the route name only when the request omitted a model id.
        ledger_model = model if model not in (None, "", "unknown") else route.name

        fwd_headers = [(k, v) for k, v in request.headers.items() if k.lower() not in _DROP]

        if serialize_lock is not None:
            await serialize_lock.acquire()   # serialize so per-request energy windows can't overlap
        label = _label_factory()
        meter.begin(label)
        idle.request_started()
        in_flight = idle.in_flight()         # concurrency at start (== 1 when serialized / measured alone)
        t0 = time.monotonic()
        ts_start = time.time()

        logger.info("req.start", extra=event(request_id=label, model=ledger_model, req_type=req_type,
                                             stream=is_stream, upstream=route.upstream,
                                             in_flight=in_flight, body_bytes=len(raw)))

        try:
            up_req = client.build_request(
                "POST", f"{route.upstream}/v1/{path}", content=raw, headers=fwd_headers
            )
            up_resp = await client.send(up_req, stream=True)
        except Exception as e:
            meter.end(label)
            idle.request_finished()
            if serialize_lock is not None:
                serialize_lock.release()
            logger.error("req.upstream_error", extra=event(request_id=label, model=ledger_model,
                                                           error_type=type(e).__name__, error=str(e)[:300]))
            return JSONResponse(
                {"error": {"message": "upstream request failed", "type": "upstream_error"}},
                status_code=502,
            )

        counter = SelfCounter(req_json) if is_stream else None
        captured = bytearray() if not is_stream else None
        first_chunk_t: list[float] = []   # records monotonic time of the first streamed chunk
        aborted: list = []

        def _finalize(usage: TokenUsage) -> None:
            try:
                idle.request_finished()
                window = meter.end(label)
                dt = time.monotonic() - t0
                idle_e = idle.energy_over(dt)
                marginal_j = (window - idle_e).total_j
                ttft = (first_chunk_t[0] - t0) if first_chunk_t else None
                cold = detector.observe(ledger_model, ttft, marginal_j, dt)
                if cold.is_cold and cold.load_energy_j > 0:
                    ledger.insert_model_load(ts=ts_start, model=ledger_model, upstream=route.upstream,
                                             load_energy_j=cold.load_energy_j,
                                             duration_ms=cold.load_time_s * 1000.0, trigger=cold.trigger)
                    marginal_j = max(0.0, marginal_j - cold.load_energy_j)
                kwh = marginal_j / 3.6e6
                cost = rate.price(kwh)
                ledger.insert(LedgerRow(
                    ts_start=ts_start, ts_end=time.time(), model=ledger_model, req_type=req_type,
                    e_window_j=window.total_j, e_idle_j=idle_e.total_j,
                    e_marginal_j=marginal_j, kwh_marginal=kwh,
                    rate_usd_kwh=rate.usd_per_kwh, cost_marginal_usd=cost,
                    tok_in=usage.input if usage else None, tok_out=usage.output if usage else None,
                    tok_source=usage.source if usage else "none",
                    energy_confidence="estimated (±15-30%)" if usage and usage.source != "none" else "energy-only",
                    cold=cold.is_cold, in_flight=in_flight, request_id=label,
                ))
                logger.info("req.finish", extra=event(
                    request_id=label, model=ledger_model, duration_s=round(dt, 2),
                    ttft_s=round(ttft, 2) if ttft is not None else None, status=up_resp.status_code,
                    tok_in=usage.input if usage else None, tok_out=usage.output if usage else None,
                    tok_source=usage.source if usage else "none", marginal_j=round(marginal_j, 1),
                    kwh=kwh, cost=cost, cold=cold.is_cold,
                    aborted=(aborted[0] if aborted else None)))
            finally:
                if serialize_lock is not None:
                    serialize_lock.release()

        resp_headers = [(k, v) for k, v in up_resp.headers.items() if k.lower() not in _DROP]

        if is_stream:
            async def body_iter():
                n = 0; nb = 0
                try:
                    async for chunk in up_resp.aiter_raw():
                        if not first_chunk_t:
                            first_chunk_t.append(time.monotonic())
                        n += 1; nb += len(chunk)
                        counter.feed(chunk)
                        yield chunk
                except (GeneratorExit, asyncio.CancelledError):
                    aborted.append("client_disconnect")
                    logger.info("req.client_disconnect", extra=event(
                        request_id=label, model=ledger_model, chunks=n, bytes=nb,
                        elapsed_s=round(time.monotonic() - t0, 2)))
                    raise
                except Exception as e:
                    aborted.append("stream_error")
                    logger.warning("req.stream_error", extra=event(
                        request_id=label, model=ledger_model, error_type=type(e).__name__,
                        error=str(e)[:300], chunks=n, bytes=nb,
                        elapsed_s=round(time.monotonic() - t0, 2)))
                    raise
                finally:
                    await up_resp.aclose()
                    _finalize(counter.result())
            return StreamingResponse(body_iter(), status_code=up_resp.status_code,
                                     headers=dict(resp_headers))
        else:
            n = 0; nb = 0
            try:
                async for chunk in up_resp.aiter_raw():
                    n += 1; nb += len(chunk)
                    captured.extend(chunk)
            except Exception as e:
                aborted.append("stream_error")
                logger.warning("req.stream_error", extra=event(
                    request_id=label, model=ledger_model, error_type=type(e).__name__,
                    error=str(e)[:300], chunks=n, bytes=nb, elapsed_s=round(time.monotonic() - t0, 2)))
                raise
            finally:
                await up_resp.aclose()
                body = bytes(captured)
                try:
                    usage = usage_from_response_json(json.loads(body))
                except json.JSONDecodeError:
                    usage = None
                _finalize(usage or TokenUsage(None, None, None, "none", "energy-only"))
            return Response(content=body, status_code=up_resp.status_code,
                            headers=dict(resp_headers))

    return Starlette(routes=[Route("/v1/{path:path}", forward, methods=["POST"])], lifespan=lifespan)
