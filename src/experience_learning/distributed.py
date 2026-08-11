from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")


def broadcast_object(value: T | None, *, source: int = 0) -> T:
    try:
        import torch.distributed as dist
    except ImportError:
        if value is None:
            raise RuntimeError("single-process broadcast received no value") from None
        return value

    if not dist.is_available() or not dist.is_initialized():
        if value is None:
            raise RuntimeError("single-process broadcast received no value")
        return value
    packet: list[Any] = [value]
    dist.broadcast_object_list(packet, src=source)
    return packet[0]


def all_gather_objects(value: T) -> list[T]:
    try:
        import torch.distributed as dist
    except ImportError:
        return [value]

    if not dist.is_available() or not dist.is_initialized():
        return [value]
    gathered: list[T | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return [item for item in gathered if item is not None]
