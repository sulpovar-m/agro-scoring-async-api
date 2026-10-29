from contextlib import asynccontextmanager

import aiosqlite

from app.errors import ServiceError

COLUMNS = (
    "request_id",
    "farm_id",
    "risk_score",
    "risk_level",
    "recommendation",
    "model_version",
    "source",
)


class PredictionRepository:
    def __init__(self, path):
        self.path = path

    @asynccontextmanager
    async def connection(self):
        try:
            async with aiosqlite.connect(self.path, timeout=5) as db:
                db.row_factory = aiosqlite.Row
                yield db
        except aiosqlite.Error as error:
            raise ServiceError(503, "Prediction storage is unavailable") from error

    async def initialize(self):
        async with self.connection() as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("""CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL, farm_id TEXT NOT NULL,
                risk_score REAL NOT NULL, risk_level TEXT NOT NULL,
                recommendation TEXT NOT NULL, model_version TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            )""")
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_request ON predictions(request_id)"
            )
            await db.commit()

    async def save_many(self, predictions, source):
        rows = [
            tuple(p.model_dump()[key] if key != "source" else source for key in COLUMNS)
            for p in predictions
        ]
        async with self.connection() as db:
            await db.executemany(
                "INSERT INTO predictions ("
                + ",".join(COLUMNS)
                + ") VALUES (?,?,?,?,?,?,?)",
                rows,
            )
            await db.commit()

    async def recent(self, limit, risk_level):
        query = "SELECT * FROM predictions"
        args = []
        if risk_level:
            query += " WHERE risk_level = ?"
            args.append(risk_level)
        query += " ORDER BY id DESC LIMIT ?"
        async with self.connection() as db:
            cursor = await db.execute(query, [*args, limit])
            return [dict(row) for row in await cursor.fetchall()]

    async def by_request(self, request_id):
        async with self.connection() as db:
            cursor = await db.execute(
                "SELECT * FROM predictions WHERE request_id = ? ORDER BY id",
                (request_id,),
            )
            return [dict(row) for row in await cursor.fetchall()]

    async def by_weekday(self, weekday_number, page, per_page):
        where = "strftime('%w', created_at) = ?"
        async with self.connection() as db:
            # COUNT и страница читаются из одного снимка БД.
            await db.execute("BEGIN")
            cursor = await db.execute(
                "SELECT COUNT(*) FROM predictions WHERE " + where, (weekday_number,)
            )
            total = (await cursor.fetchone())[0]
            cursor = await db.execute(
                "SELECT * FROM predictions WHERE "
                + where
                + " ORDER BY id DESC LIMIT ? OFFSET ?",
                (weekday_number, per_page, (page - 1) * per_page),
            )
            return [dict(row) for row in await cursor.fetchall()], total
