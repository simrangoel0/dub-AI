from __future__ import annotations
import json 

from datetime import datetime
from typing import Optional, Any, Dict, List

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    Float,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from sqlalchemy.exc import IntegrityError

from backend.core.models import ContextSelectionResult

# ------------------------------------------------------------------
# Engine & Session
# ------------------------------------------------------------------
DATABASE_URL = "sqlite:///./glassbox.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + threads
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()


# ------------------------------------------------------------------
# ORM Models
# ------------------------------------------------------------------

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, index=True, nullable=False)  # e.g. "default"
    role = Column(String, nullable=False)                         # "user", "assistant", etc.
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Run(Base):
    __tablename__ = "runs"

    run_id = Column(String, primary_key=True)
    message_id = Column(String, nullable=False)
    conversation_id = Column(String, nullable=False, index=True)
    user_query = Column(Text, nullable=False)
    run_label = Column(String, nullable=True)      # <--- ADD THIS
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    top_k = Column(Integer)
    selection_mode = Column(String)
    trace_file = Column(String)
    repo_snapshot = Column(Text)

    chunks = relationship("RunChunk", back_populates="run", cascade="all, delete-orphan")
    attributions = relationship("AttributionRow", back_populates="run", cascade="all, delete-orphan")


class ChunkModel(Base):
    """
    Global chunk registry (code + optional chat).
    """
    __tablename__ = "chunks"

    chunk_id = Column(String, primary_key=True)  # e.g. "auth.py_2"
    source = Column(String, nullable=False)      # "code" | "chat"
    file_path = Column(String, nullable=False)
    start_line = Column(Integer)
    end_line = Column(Integer)
    text = Column(Text, nullable=False)
    meta_json = Column(Text)                     # JSON string of Chunk.meta (optional)


class RunChunk(Base):
    """
    Per-run view of a chunk: selected vs dropped, scores, rationale.
    """
    __tablename__ = "run_chunks"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, ForeignKey("runs.run_id"), nullable=False)
    chunk_id = Column(String, nullable=False)

    file_path = Column(String, nullable=False)
    source = Column(String, nullable=False)          # "code" | "chat"
    start_line = Column(Integer)
    end_line = Column(Integer)
    text = Column(Text, nullable=False)

    similarity_score = Column(Float)
    relevance_score = Column(Float)
    is_selected = Column(Boolean, nullable=False)
    selection_rationale = Column(Text)

    run = relationship("Run", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("run_id", "chunk_id", name="uq_run_chunk"),
    )


class AttributionRow(Base):
    """
    Per-run, per-chunk attribution score + explanation.
    """
    __tablename__ = "attributions"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, ForeignKey("runs.run_id"), nullable=False)
    chunk_id = Column(String, nullable=False)

    influence_score = Column(Float, nullable=False)
    explanation = Column(Text)
    evidence_json = Column(Text)  # JSON list of quotes

    run = relationship("Run", back_populates="attributions")

    __table_args__ = (
        UniqueConstraint("run_id", "chunk_id", name="uq_attr_run_chunk"),
    )


class ContextModification(Base):
    """
    User boosts / excludes chunks for a given run.
    """
    __tablename__ = "context_modifications"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, ForeignKey("runs.run_id"), nullable=False)
    chunk_id = Column(String, nullable=False)
    action = Column(String, nullable=False)     # "boost" | "exclude" | "pin" | ...
    reason = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def seed_chunks_from_index(index) -> None:
    """
    index: an instance of CodebaseIndex (has .chunks List[Chunk])
    Inserts all code chunks into the 'chunks' table if not already present.
    """
    from backend.core.models import Chunk

    db = SessionLocal()
    try:
        for c in index.chunks:
            # Only seed code chunks here; chat chunks come per-run if you want.
            if c.source != "code":
                continue

            # Try insert; if chunk_id already exists, ignore.
            obj = ChunkModel(
                chunk_id=c.chunk_id,
                source=c.source,
                file_path=c.file_path,
                start_line=c.start_line,
                end_line=c.end_line,
                text=c.text,
                meta_json=json.dumps(c.meta or {}),
            )
            db.add(obj)
            try:
                db.commit()
            except IntegrityError:
                db.rollback()  # chunk_id already exists, that's fine
    finally:
        db.close()


def persist_run(
    run_id: str,
    message_id: str,
    conversation_id: str,
    user_query: str,
    run_label: Optional[str],
    ctx: ContextSelectionResult,
    answer_result: Dict[str, Any],
    attribution_result: Dict[str, Any],
    trace_file: str | None,
) -> None:
    """
    Persist a completed pipeline run into the DB.

    - Inserts Run
    - Inserts RunChunks (selected + dropped)
    - Inserts Attributions
    """
    db: Session = SessionLocal()
    try:
        # ---- Run row ----
        run = Run(
            run_id=run_id,
            message_id=message_id,
            conversation_id=conversation_id,
            user_query=user_query,
            run_label=run_label,
            top_k=ctx.top_k,
            selection_mode=ctx.meta.get("selection_mode"),
            trace_file=trace_file,
            repo_snapshot=None,  # you can dump repo_state JSON here later
        )
        db.add(run)

        # ---- RunChunks ----
        for sc in ctx.selected_chunks + ctx.dropped_chunks:
            c = sc.chunk
            rc = RunChunk(
                run_id=run_id,
                chunk_id=c.chunk_id,
                file_path=c.file_path,
                source=c.source,
                start_line=c.start_line,
                end_line=c.end_line,
                text=c.text,
                similarity_score=sc.similarity_score,
                relevance_score=sc.relevance_score,
                is_selected=(sc in ctx.selected_chunks),
                selection_rationale=sc.rationale,
            )
            db.add(rc)

        # ---- Attributions ----
        # We expect attribution_result["explanations"] as {chunk_id: {explanation, evidence}}
        explanations = attribution_result.get("explanations", {}) or {}
        influence_scores = attribution_result.get("influence_scores", {}) or {}

        for cid, score in influence_scores.items():
            expl = explanations.get(cid, {})
            ar = AttributionRow(
                run_id=run_id,
                chunk_id=cid,
                influence_score=float(score),
                explanation=expl.get("explanation"),
                evidence_json=json.dumps(expl.get("evidence", [])),
            )
            db.add(ar)

        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def persist_chat_messages(conversation_id: str, history: List[Dict[str, Any]]) -> None:
    db = SessionLocal()
    try:
        for msg in history:
            cm = ChatMessage(
                conversation_id=conversation_id,
                role=msg.get("role", "user"),
                content=msg.get("content") or "",
            )
            db.add(cm)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# ------------------------------------------------------------------
# DB init helper
# ------------------------------------------------------------------

def init_db() -> None:
    Base.metadata.create_all(bind=engine)

init_db()