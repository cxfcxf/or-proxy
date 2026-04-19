import asyncio
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ProxyState:
    ranked_models: list[str] = field(default_factory=list)
    last_refresh: datetime | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


state = ProxyState()
