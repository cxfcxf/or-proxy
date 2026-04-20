import asyncio
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ProxyState:
    ranked_models: list[dict] = field(default_factory=list)  # [{"id": ..., "context_length": ...}]
    last_refresh: datetime | None = None
    sticky_model: str | None = None  # last successful model id
    sticky_since: datetime | None = None  # when sticky was last set
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


state = ProxyState()
