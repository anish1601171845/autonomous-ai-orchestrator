"""Controlled fault injection for testing and demonstrating self-healing. Faults only fire inside an
explicit `inject(...)` context (or via the API demo endpoint); nothing is injected in normal runs.
Every injected fault is labelled `simulated` in the recovery_events table.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field


class SimulatedFault(Exception):
    simulated = True


class SimulatedLLMTimeout(SimulatedFault, TimeoutError):
    pass


class SimulatedRateLimit(SimulatedFault):
    def __init__(self, msg: str = "simulated HTTP 429 rate limit", retry_after: float = 0.05):
        super().__init__(msg)
        self.retry_after = retry_after


class SimulatedNetworkError(SimulatedFault, ConnectionError):
    pass


class SimulatedAgentCrash(SimulatedFault):
    """The assigned agent process died / is unavailable. Retrying on the SAME agent keeps failing."""


class SimulatedWorkerCrash(SimulatedFault):
    """The orchestrator worker crashed mid-workflow (aborts the whole graph run)."""


class SimulatedDBOutage(SimulatedFault):
    pass


SCENARIOS = {
    # name: (injection point, exception factory, default number of firings)
    "api_timeout": ("execute", lambda: SimulatedLLMTimeout("simulated LLM API timeout"), 2),
    "rate_limit": ("execute", lambda: SimulatedRateLimit(), 2),
    "network_error": ("execute", lambda: SimulatedNetworkError("simulated connection reset"), 1),
    "agent_crash": ("execute", lambda: SimulatedAgentCrash("simulated agent crash"), 10_000),
    "db_outage": ("persist", lambda: SimulatedDBOutage("simulated database outage"), 2),
    "qa_rejection": ("qa", None, 1),
    "repeated_failure": ("execute", lambda: SimulatedLLMTimeout("simulated persistent timeout"), 10_000),
    "worker_crash": ("node", lambda: SimulatedWorkerCrash("simulated worker crash"), 1),
}


@dataclass
class FaultSpec:
    scenario: str
    remaining: int
    agent: str | None = None          # only fire for this agent (agent_crash)
    task_id: str | None = None        # only fire for this task
    node: str | None = "execute"      # graph node for worker_crash
    fired: int = 0
    fired_on: list[str] = field(default_factory=list)


class FaultInjector:
    def __init__(self) -> None:
        self._specs: list[FaultSpec] = []
        self._lock = threading.Lock()

    def active(self) -> bool:
        return bool(self._specs)

    def add(self, spec: FaultSpec) -> FaultSpec:
        with self._lock:
            self._specs.append(spec)
        return spec

    def clear(self) -> None:
        with self._lock:
            self._specs.clear()

    def check(self, point: str, *, agent: str | None = None, task_id: str | None = None, node: str | None = None) -> None:
        """Raise the configured simulated fault if one matches this point."""
        with self._lock:
            for s in self._specs:
                p, factory, _ = SCENARIOS[s.scenario]
                if p != point or s.remaining <= 0 or factory is None:
                    continue
                if s.agent and s.agent != agent:
                    continue
                if s.task_id and s.task_id != task_id:
                    continue
                if point == "node" and s.node and s.node != node:
                    continue
                s.remaining -= 1
                s.fired += 1
                s.fired_on.append(task_id or node or point)
                raise factory()

    def force_qa_rejection(self, task_id: str | None) -> bool:
        with self._lock:
            for s in self._specs:
                if s.scenario == "qa_rejection" and s.remaining > 0 and (not s.task_id or s.task_id == task_id):
                    s.remaining -= 1
                    s.fired += 1
                    s.fired_on.append(task_id or "qa")
                    return True
        return False


INJECTOR = FaultInjector()


@contextmanager
def inject(scenario: str, times: int | None = None, *, agent: str | None = None, task_id: str | None = None, node: str | None = "execute"):
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
    spec = INJECTOR.add(FaultSpec(scenario, SCENARIOS[scenario][2] if times is None else times, agent, task_id, node))
    try:
        yield spec
    finally:
        with INJECTOR._lock:
            if spec in INJECTOR._specs:
                INJECTOR._specs.remove(spec)
