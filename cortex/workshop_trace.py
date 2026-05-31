"""Direct OpenTelemetry tracing into Raindrop Workshop (no API key required).

The Raindrop SDK gates span/signal export on a cloud write key. But Workshop itself
ingests raw OTLP JSON at http://localhost:5899/v1/traces. So we send the agent's full
internal trajectory — design, hypotheses, tool calls, interpretations, caught errors,
and self-corrections — straight to Workshop as nested spans, bypassing the key gate.

Everything here is guarded: if Workshop/OTel is unavailable, it silently no-ops.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

_PROVIDER = None
_TRACER = None


def _setup():
    global _PROVIDER, _TRACER
    if _TRACER is not None:
        return _TRACER
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        endpoint = os.getenv("CORTEX_WORKSHOP_OTLP", "http://localhost:5899/v1/traces")
        prov = TracerProvider(resource=Resource.create({"service.name": "cortex"}))
        prov.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        _PROVIDER = prov
        _TRACER = prov.get_tracer("cortex")
    except Exception:
        _TRACER = False
    return _TRACER


class WorkshopTrace:
    """A single run trace. Children are explicitly parented (robust across async/threads)."""

    def __init__(self, event_name: str, input_text: str):
        self.ok = False
        self.root = None
        tracer = _setup()
        if not tracer:
            return
        try:
            from opentelemetry import trace
            self._trace = trace
            self.tracer = tracer
            self.root = tracer.start_span(event_name)
            self.root.set_attribute("input", (input_text or "")[:2000])
            self.root_ctx = trace.set_span_in_context(self.root)
            self.ok = True
        except Exception:
            self.ok = False

    def _set_attrs(self, span, attrs: dict):
        for k, v in attrs.items():
            try:
                span.set_attribute(k, v if isinstance(v, (int, float, bool)) else str(v)[:1500])
            except Exception:
                pass

    @contextmanager
    def span(self, name: str, parent=None, **attrs):
        """Open a child span (under `parent` span, or the run root). Yields the span."""
        if not self.ok:
            yield None
            return
        ctx = self._trace.set_span_in_context(parent) if parent is not None else self.root_ctx
        s = self.tracer.start_span(name, context=ctx)
        self._set_attrs(s, attrs)
        try:
            yield s
        except Exception as e:
            try:
                s.record_exception(e)
            except Exception:
                pass
            raise
        finally:
            s.end()

    def signal(self, name: str, parent=None, sentiment: str = "", reason: str = "", **attrs):
        """Record a caught signal / self-correction as a short, clearly-named span."""
        if not self.ok:
            return
        prefix = "✓" if sentiment == "POSITIVE" else "⚠" if sentiment == "NEGATIVE" else "•"
        ctx = self._trace.set_span_in_context(parent) if parent is not None else self.root_ctx
        s = self.tracer.start_span(f"{prefix} signal: {name}", context=ctx)
        self._set_attrs(s, {"signal.name": name, "signal.sentiment": sentiment,
                            "signal.reason": reason, **attrs})
        s.end()

    def add_output(self, output: str):
        if self.ok and output:
            try:
                self.root.set_attribute("output", str(output)[:4000])
            except Exception:
                pass

    def finish(self, output: str = ""):
        if not self.ok:
            return
        try:
            self.add_output(output)
            self.root.end()
            if _PROVIDER:
                _PROVIDER.force_flush()
        except Exception:
            pass
