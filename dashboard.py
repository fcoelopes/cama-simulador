"""Queries de métricas de desempenho a partir da tabela `attempts`."""
from __future__ import annotations
import sqlite3
from dataclasses import dataclass

from config import QUESTIONS_DB


def _conn():
    return sqlite3.connect(QUESTIONS_DB)


@dataclass
class OverallStats:
    total_attempts: int
    total_correct: int
    accuracy: float           # 0..1
    sessions: int
    unique_questions: int


def overall_stats() -> OverallStats:
    with _conn() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(is_correct), 0) AS correct,
                COUNT(DISTINCT session_id) AS sessions,
                COUNT(DISTINCT question_id) AS unique_qs
            FROM attempts
        """).fetchone()
    total, correct, sessions, unique_qs = row
    acc = correct / total if total else 0.0
    return OverallStats(total, correct, acc, sessions, unique_qs)


def by_topic() -> list[dict]:
    """Acerto por tópico, ordenado pelo pior desempenho primeiro."""
    with _conn() as conn:
        rows = conn.execute("""
            SELECT
                topic,
                COUNT(*) AS total,
                SUM(is_correct) AS correct
            FROM attempts
            GROUP BY topic
            HAVING total >= 1
            ORDER BY (CAST(SUM(is_correct) AS REAL) / COUNT(*)) ASC
        """).fetchall()
    return [
        {"topic": t, "total": tot, "correct": c, "accuracy": c / tot if tot else 0}
        for t, tot, c in rows
    ]


def by_session() -> list[dict]:
    """Desempenho por sessão de simulado, em ordem cronológica."""
    with _conn() as conn:
        rows = conn.execute("""
            SELECT
                session_id,
                MIN(answered_at) AS started,
                COUNT(*) AS total,
                SUM(is_correct) AS correct
            FROM attempts
            GROUP BY session_id
            ORDER BY started ASC
        """).fetchall()
    return [
        {
            "session_id": sid,
            "started": started,
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0,
        }
        for sid, started, total, correct in rows
    ]


def hardest_questions(limit: int = 10) -> list[dict]:
    """Questões com mais tentativas e pior taxa de acerto."""
    with _conn() as conn:
        rows = conn.execute("""
            SELECT
                a.question_id,
                q.stem,
                q.topic,
                COUNT(*) AS attempts,
                SUM(a.is_correct) AS correct
            FROM attempts a
            JOIN questions q ON q.id = a.question_id
            GROUP BY a.question_id
            HAVING attempts >= 1
            ORDER BY (CAST(SUM(a.is_correct) AS REAL) / COUNT(*)) ASC, attempts DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [
        {
            "question_id": qid,
            "stem": stem,
            "topic": topic,
            "attempts": att,
            "correct": correct,
            "accuracy": correct / att if att else 0,
        }
        for qid, stem, topic, att, correct in rows
    ]


def reset_history() -> None:
    """Apaga todo o histórico de tentativas (mantém o cache de questões)."""
    with _conn() as conn:
        conn.execute("DELETE FROM attempts")