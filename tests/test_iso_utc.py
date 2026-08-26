"""``_iso_utc`` 时间序列化测试：SQLite 往返后 naive UTC 必须补 ``+00:00``。

背景：``CacheEvent`` / ``CostSupplement`` 的 ``created_at`` 以 aware UTC 写入，
但 SQLite DATETIME 存储格式不含时区偏移，读回为 naive。前端 ``new Date()``
对无偏移 ISO 串按本地时区解析，会差出一个时区偏移量（UTC+8 即 8 小时）。
"""

from datetime import UTC, datetime

from cost_control.web_api import _iso_utc


def test_iso_utc_none_returns_none():
    assert _iso_utc(None) is None


def test_iso_utc_naive_gets_utc_suffix():
    out = _iso_utc(datetime(2026, 8, 25, 3, 0, 0))
    assert out == "2026-08-25T03:00:00+00:00"


def test_iso_utc_aware_kept_asis():
    dt = datetime(2026, 8, 25, 3, 0, 0, tzinfo=UTC)
    assert _iso_utc(dt) == "2026-08-25T03:00:00+00:00"


async def test_cache_event_roundtrip_keeps_instant():
    """aware UTC 写入 → SQLite 读回 naive → ``_iso_utc`` 还原同一时刻。

    这是缓存破坏事件时间的回归测试：修复前读回值裸 ``isoformat()`` 无偏移，
    前端按本地时区解析导致显示时间偏差一个时区偏移量。
    """
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlmodel import SQLModel, select

    from cost_control.store import CacheEvent

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda c: SQLModel.metadata.create_all(
                c,
                tables=[CacheEvent.__table__],
                checkfirst=True,
            )
        )
    written = datetime(2026, 8, 25, 3, 0, 0, tzinfo=UTC)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        session.add(CacheEvent(umo="u", type="context_reset", created_at=written))
        await session.commit()
        row = (await session.execute(select(CacheEvent))).scalars().first()
        # SQLite 存储格式丢 tzinfo（机制本身），序列化必须补回
        assert row.created_at.tzinfo is None
        out = _iso_utc(row.created_at)
        assert out is not None and out.endswith("+00:00")
        assert datetime.fromisoformat(out) == written
    await engine.dispose()
