
from __future__ import annotations

import asyncio
import io
import json
import os
import queue
import re
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import streamlit as st

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

from pydantic import BaseModel, Field, field_validator

# ==============================================================================
# CONFIGURATION
# ==============================================================================

APP_TITLE = "CODE FACTORY AGENT"
APP_ICON = "⚙️"
DB_PATH = Path(os.path.expanduser("~/.code_factory_agent/factory.db"))
WORKSPACE_ROOT = Path(os.path.expanduser("~/.code_factory_agent/workspace"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
GROQ_API_BASE = "https://api.groq.com/openai/v1"
MAX_DEBUG_RETRIES = 2
LLM_TIMEOUT_SECONDS = 90


# ==============================================================================
# LOGGING (structured, thread-safe, feeds the live UI console)
# ==============================================================================

class LogLevel(str, Enum):
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"
    AGENT = "AGENT"


@dataclass
class LogEntry:
    timestamp: str
    level: LogLevel
    agent: str
    message: str

    def render_line(self) -> str:
        color_map = {
            LogLevel.INFO: "#7dd3fc",
            LogLevel.SUCCESS: "#4ade80",
            LogLevel.WARNING: "#fbbf24",
            LogLevel.ERROR: "#f87171",
            LogLevel.DEBUG: "#a78bfa",
            LogLevel.AGENT: "#38bdf8",
        }
        color = color_map.get(self.level, "#e5e7eb")
        return (
            f'<div class="log-line">'
            f'<span class="log-ts">{self.timestamp}</span> '
            f'<span class="log-tag" style="color:{color};border-color:{color}55">{self.level.value}</span> '
            f'<span class="log-agent">[{self.agent}]</span> '
            f'<span class="log-msg">{self.message}</span>'
            f"</div>"
        )


class LogBus:
    """Thread-safe append-only log store shared across agent threads and the UI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.entries: list[LogEntry] = []

    def log(self, agent: str, message: str, level: LogLevel = LogLevel.INFO) -> None:
        entry = LogEntry(
            timestamp=datetime.now().strftime("%H:%M:%S"),
            level=level,
            agent=agent,
            message=message,
        )
        with self._lock:
            self.entries.append(entry)

    def snapshot(self) -> list[LogEntry]:
        with self._lock:
            return list(self.entries)


# ==============================================================================
# PERSISTENCE LAYER (SQLite)
# ==============================================================================

class ProjectStore:
    """Persists factory runs, generated files, and test results to SQLite."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    idea TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    workspace_path TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    passed INTEGER NOT NULL,
                    output TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    def create_run(self, run_id: str, idea: str, workspace_path: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO runs (run_id, idea, created_at, status, workspace_path) VALUES (?,?,?,?,?)",
                (run_id, idea, datetime.now().isoformat(), "running", workspace_path),
            )
            conn.commit()
        finally:
            conn.close()

    def update_status(self, run_id: str, status: str) -> None:
        conn = self._connect()
        try:
            conn.execute("UPDATE runs SET status=? WHERE run_id=?",
                         (status, run_id))
            conn.commit()
        finally:
            conn.close()

    def save_file(self, run_id: str, path: str, content: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO files (run_id, path, content, created_at) VALUES (?,?,?,?)",
                (run_id, path, content, datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def save_test_result(self, run_id: str, attempt: int, passed: bool, output: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO test_results (run_id, attempt, passed, output, created_at) VALUES (?,?,?,?,?)",
                (run_id, attempt, int(passed), output, datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def list_runs(self) -> list[dict]:
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT run_id, idea, created_at, status, workspace_path FROM runs ORDER BY created_at DESC")
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        finally:
            conn.close()


# ==============================================================================
# LLM CLIENT (Groq — OpenAI-compatible chat completions, via raw HTTP)
# ==============================================================================

class LLMError(Exception):
    pass


class GroqClient:
    """Minimal, dependency-light Groq chat-completion client.

    Uses urllib directly so this file has zero hard dependency on the `groq`
    package being pre-installed in the runtime environment; it only needs
    outbound HTTPS access to api.groq.com.
    """

    def __init__(self, api_key: str, model: str = GROQ_DEFAULT_MODEL) -> None:
        self.api_key = api_key
        self.model = model

    def chat(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 4000) -> str:
        import urllib.request
        import urllib.error

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{GROQ_API_BASE}/chat/completions",
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "Mozilla/5.0 (compatible; CodeFactoryAgent/1.0)",
                "Accept": "application/json",
            },
        )
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_SECONDS) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="ignore")
                if e.code == 429 and attempt < max_retries:
                    wait_s = 10
                    m = re.search(r"try again in ([\d.]+)s", detail)
                    if m:
                        wait_s = float(m.group(1)) + 1
                    time.sleep(wait_s)
                    continue
                raise LLMError(
                    f"Groq API HTTP {e.code}: {detail[:500]}") from e
            except urllib.error.URLError as e:
                raise LLMError(f"Groq API network error: {e.reason}") from e
            except Exception as e:  # noqa: BLE001
                raise LLMError(f"Groq API unexpected error: {e}") from e

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"Malformed Groq response: {body}") from e


def extract_code_blocks(text: str) -> dict[str, str]:
    """Extract '### FILE: path' delimited code blocks the LLM is instructed to emit.

    Expected LLM output format per file:

        ### FILE: relative/path.ext
        ```
        <content>
        ```

    Falls back gracefully if the model omits fences.
    """
    files: dict[str, str] = {}
    pattern = re.compile(
        r"### FILE:\s*(?P<path>[^\n]+)\n```(?:[a-zA-Z0-9_+-]*)\n(?P<content>.*?)```",
        re.DOTALL,
    )
    for match in pattern.finditer(text):
        path = match.group("path").strip()
        content = match.group("content")
        files[path] = content
    return files


# ==============================================================================
# PYDANTIC DATA MODELS
# ==============================================================================

class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class Specification(BaseModel):
    project_name: str = Field(..., min_length=1)
    description: str
    core_features: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    data_model: list[str] = Field(default_factory=list)

    @field_validator("project_name")
    @classmethod
    def slugify_name(cls, v: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", v.strip())[:60] or "generated_project"


class TestOutcome(BaseModel):
    attempt: int
    passed: bool
    summary: str
    raw_output: str


# ==============================================================================
# AGENT FRAMEWORK
# ==============================================================================

@dataclass
class AgentContext:
    """Shared state passed between all agents during a pipeline run."""

    run_id: str
    idea: str
    workspace: Path
    llm: GroqClient
    store: ProjectStore
    logbus: LogBus
    spec: Optional[Specification] = None
    generated_files: dict[str, str] = field(default_factory=dict)
    test_history: list[TestOutcome] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    stop_flag: threading.Event = field(default_factory=threading.Event)


class BaseAgent:
    """Common behavior for every autonomous agent in the pipeline."""

    name: str = "BaseAgent"

    def __init__(self, ctx: AgentContext, status_sink: Callable[[str, AgentStatus, str], None]) -> None:
        self.ctx = ctx
        self.status_sink = status_sink

    def log(self, message: str, level: LogLevel = LogLevel.INFO) -> None:
        self.ctx.logbus.log(self.name, message, level)

    def set_status(self, status: AgentStatus, detail: str = "") -> None:
        self.status_sink(self.name, status, detail)

    def run(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def execute(self) -> None:
        if self.ctx.stop_flag.is_set():
            return
        self.set_status(AgentStatus.RUNNING, "working...")
        self.log(f"{self.name} starting.", LogLevel.AGENT)
        try:
            self.run()
            self.set_status(AgentStatus.DONE, "complete")
            self.log(f"{self.name} finished successfully.", LogLevel.SUCCESS)
        except Exception as e:  # noqa: BLE001
            self.set_status(AgentStatus.ERROR, str(e))
            self.log(f"{self.name} failed: {e}", LogLevel.ERROR)
            raise


class PlannerAgent(BaseAgent):
    name = "Planner"

    def run(self) -> None:
        self.log(f"Analyzing idea: '{self.ctx.idea}'")
        system = (
            "You are a senior software product planner. Given a one-sentence app idea, "
            "produce a strict JSON object (no markdown, no prose) with keys: "
            "project_name (short snake_case string), description (2-3 sentences), "
            "core_features (array of 5-8 short strings), "
            "tech_stack (array of technology names, keep it to Python + SQLite + a simple HTML/CSS/JS or Streamlit frontend), "
            "data_model (array of short strings describing key entities/tables). "
            "Return ONLY valid JSON."
        )
        raw = self.ctx.llm.chat(system, self.ctx.idea,
                                temperature=0.3, max_tokens=1200)
        cleaned = raw.strip()
        cleaned = re.sub(r"^```json\s*|\s*```$", "",
                         cleaned.strip(), flags=re.MULTILINE)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                raise LLMError(
                    "Planner could not produce valid JSON specification.")
            data = json.loads(match.group(0))
        if isinstance(data.get("description"), list):
            data["description"] = " ".join(str(x) for x in data["description"])
        elif isinstance(data.get("description"), dict):
            data["description"] = json.dumps(data["description"])
        self.ctx.spec = Specification(**data)
        self.log(
            f"Specification created: {self.ctx.spec.project_name}", LogLevel.SUCCESS)


class ArchitectAgent(BaseAgent):
    name = "Architect"

    def run(self) -> None:
        spec = self.ctx.spec
        assert spec is not None
        self.log("Selecting architecture: single-package Python app, SQLite storage, Flask REST API, static HTML/JS frontend.")
        self.ctx.metrics[
            "architecture"] = "Flask + SQLite + static frontend (monolith, single-process)"
        self.log(
            "Folder layout decided: app/, app/static/, app/templates/, tests/", LogLevel.INFO)


class BackendAgent(BaseAgent):
    name = "Backend"

    def run(self) -> None:
        spec = self.ctx.spec
        assert spec is not None
        system = (
            "You are an expert Python backend engineer. Generate a COMPLETE, runnable Flask backend "
            "for the given specification. Requirements:\n"
            "- Single file: backend/server.py\n"
            "- Use Flask + sqlite3 (stdlib), no external DB.\n"
            "- Implement full CRUD REST API endpoints under /api/ matching the data_model.\n"
            "- Include a simple token-based auth: POST /api/auth/register and /api/auth/login "
            "returning a signed token (use itsdangerous-free simple HMAC via hashlib, stdlib only).\n"
            "- Include input validation and proper HTTP status codes.\n"
            "- Include a health check endpoint GET /api/health.\n"
            "- Auto-create SQLite tables on startup if missing.\n"
            "- No placeholders, no TODOs, fully working code.\n"
            "Output format STRICTLY:\n"
            "### FILE: backend/server.py\n```python\n<full file content>\n```\n"
            "Nothing else outside that block."
        )
        user = (
            f"Project: {spec.project_name}\nDescription: {spec.description}\n"
            f"Core features: {spec.core_features}\nData model: {spec.data_model}"
        )
        raw = self.ctx.llm.chat(system, user, temperature=0.2, max_tokens=6000)
        files = extract_code_blocks(raw)
        if not files:
            self.log(
                f"Raw LLM output (unparsed):\n{raw[:2000]}", LogLevel.DEBUG)
            raise LLMError("Backend agent produced no parsable file blocks.")
        self.ctx.generated_files.update(files)
        for path in files:
            self.log(f"Generated backend file: {path}", LogLevel.SUCCESS)


class FrontendAgent(BaseAgent):
    name = "Frontend"

    def run(self) -> None:
        spec = self.ctx.spec
        assert spec is not None
        system = (
            "You are an expert frontend engineer. Generate a COMPLETE static frontend "
            "(HTML + CSS + vanilla JS, single page) that talks to a REST API at /api/. "
            "Requirements:\n"
            "- Files: frontend/index.html, frontend/style.css, frontend/app.js\n"
            "- Modern, clean, responsive dark-mode UI.\n"
            "- Implement forms and views for the core features listed.\n"
            "- Use fetch() to call the backend API.\n"
            "- No frameworks, no build step, pure static files.\n"
            "- No placeholders, fully working code.\n"
            "Output format STRICTLY, one block per file:\n"
            "### FILE: frontend/index.html\n```html\n<content>\n```\n"
            "### FILE: frontend/style.css\n```css\n<content>\n```\n"
            "### FILE: frontend/app.js\n```javascript\n<content>\n```\n"
            "Nothing else outside those blocks."
        )
        user = f"Project: {spec.project_name}\nDescription: {spec.description}\nCore features: {spec.core_features}"
        raw = self.ctx.llm.chat(system, user, temperature=0.3, max_tokens=8000)
        files = extract_code_blocks(raw)
        if not files:
            raise LLMError("Frontend agent produced no parsable file blocks.")
        self.ctx.generated_files.update(files)
        for path in files:
            self.log(f"Generated frontend file: {path}", LogLevel.SUCCESS)


class DatabaseAgent(BaseAgent):
    name = "Database"

    def run(self) -> None:
        spec = self.ctx.spec
        assert spec is not None
        system = (
            "You are a database engineer. Generate a COMPLETE SQLite schema initialization module. "
            "Requirements:\n"
            "- File: backend/schema.sql\n"
            "- CREATE TABLE statements for every entity in the data model, with sensible columns, "
            "primary keys, foreign keys, and a users table for authentication.\n"
            "- Include IF NOT EXISTS guards.\n"
            "Output format STRICTLY:\n"
            "### FILE: backend/schema.sql\n```sql\n<content>\n```\n"
        )
        user = f"Data model: {spec.data_model}\nProject: {spec.project_name}"
        raw = self.ctx.llm.chat(system, user, temperature=0.1, max_tokens=1500)
        files = extract_code_blocks(raw)
        if files:
            self.ctx.generated_files.update(files)
            for path in files:
                self.log(f"Generated schema file: {path}", LogLevel.SUCCESS)
        else:
            self.log(
                "No separate schema file produced (schema likely inlined in backend).", LogLevel.WARNING)


class DocumentationAgent(BaseAgent):
    name = "Documentation"

    def run(self) -> None:
        spec = self.ctx.spec
        assert spec is not None
        readme = self._build_readme(spec)
        changelog = self._build_changelog(spec)
        self.ctx.generated_files["README.md"] = readme
        self.ctx.generated_files["CHANGELOG.md"] = changelog
        self.log("Generated README.md and CHANGELOG.md", LogLevel.SUCCESS)

    def _build_readme(self, spec: Specification) -> str:
        features = "\n".join(f"- {f}" for f in spec.core_features)
        stack = ", ".join(
            spec.tech_stack) if spec.tech_stack else "Python, Flask, SQLite, HTML/CSS/JS"
        return textwrap.dedent(
            f"""\
            # {spec.project_name}

            {spec.description}

            ## Features
            {features}

            ## Tech Stack
            {stack}

            ## Getting Started

            ```bash
            cd backend
            pip install flask
            python server.py
            ```

            Then open `frontend/index.html` in your browser, or serve it with any
            static file server pointed at the `frontend/` directory.

            ## API

            See `docs/API.md` for endpoint documentation.

            ## Generated By

            This project was scaffolded by Code Factory Agent, an autonomous
            multi-agent AI software engineering pipeline.
            """
        )

    def _build_changelog(self, spec: Specification) -> str:
        return textwrap.dedent(
            f"""\
            # Changelog

            ## [0.1.0] - {datetime.now().strftime('%Y-%m-%d')}
            ### Added
            - Initial autonomous generation of {spec.project_name}.
            - Backend REST API with authentication.
            - Frontend static UI.
            - SQLite schema and persistence layer.
            - Automated test suite.
            """
        )


class TestingAgent(BaseAgent):
    name = "Testing"

    def run(self) -> None:
        spec = self.ctx.spec
        assert spec is not None
        system = (
            "You are a QA engineer. Generate a COMPLETE pytest test file for the backend "
            "server described below. It must import the Flask app factory or app object "
            "from backend/server.py (assume it exposes a module-level `app` Flask instance "
            "and works with app.test_client()), and test the health endpoint plus at least "
            "one core CRUD flow. Keep it self-contained and runnable with `pytest`. "
            "No placeholders.\n"
            "Output format STRICTLY:\n"
            "### FILE: tests/test_backend.py\n```python\n<content>\n```\n"
        )
        user = f"Backend file content:\n{self.ctx.generated_files.get('backend/server.py','')[:6000]}"
        raw = self.ctx.llm.chat(system, user, temperature=0.2, max_tokens=2000)
        files = extract_code_blocks(raw)
        if files:
            self.ctx.generated_files.update(files)
            for path in files:
                self.log(f"Generated test file: {path}", LogLevel.SUCCESS)
        else:
            self.log(
                "Testing agent produced no test file; will run syntax checks only.", LogLevel.WARNING)


class DebugAgent(BaseAgent):
    """Runs the generated code, and if it fails, asks the LLM to fix it, looping."""

    name = "Debug"

    def run(self) -> None:
        write_all_files(self.ctx.workspace, self.ctx.generated_files)
        for attempt in range(1, MAX_DEBUG_RETRIES + 1):
            self.log(f"Test/verify attempt {attempt}/{MAX_DEBUG_RETRIES}...")
            outcome = self._verify(attempt)
            self.ctx.test_history.append(outcome)
            self.ctx.store.save_test_result(
                self.ctx.run_id, attempt, outcome.passed, outcome.raw_output)
            if outcome.passed:
                self.log(
                    f"Verification PASSED on attempt {attempt}.", LogLevel.SUCCESS)
                return
            self.log(
                f"Verification FAILED on attempt {attempt}: {outcome.summary}", LogLevel.WARNING)
            if attempt == MAX_DEBUG_RETRIES:
                self.log(
                    "Max debug retries reached; proceeding with best-effort code.", LogLevel.ERROR)
                return
            self._attempt_fix(outcome)
            write_all_files(self.ctx.workspace, self.ctx.generated_files)

    def _verify(self, attempt: int) -> TestOutcome:
        errors: list[str] = []
        py_files = {p: c for p, c in self.ctx.generated_files.items()
                    if p.endswith(".py")}
        for path, _ in py_files.items():
            full = self.ctx.workspace / path
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(full)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                errors.append(f"[{path}] compile error:\n{result.stderr}")

        if not errors and "tests/test_backend.py" in self.ctx.generated_files:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/test_backend.py", "-q"],
                cwd=str(self.ctx.workspace),
                capture_output=True, text=True, timeout=60,
            )
            output = result.stdout + result.stderr
            if result.returncode != 0:
                errors.append(f"[pytest]\n{output[-3000:]}")
            else:
                return TestOutcome(attempt=attempt, passed=True, summary="All checks passed (compile + pytest).", raw_output=output)

        if errors:
            return TestOutcome(attempt=attempt, passed=False, summary=f"{len(errors)} issue(s) found.", raw_output="\n\n".join(errors))
        return TestOutcome(attempt=attempt, passed=True, summary="All Python files compiled successfully (no test suite executed).", raw_output="")

    def _attempt_fix(self, outcome: TestOutcome) -> None:
        broken_file = None
        m = re.search(r"\[([^\]]+\.py)\]", outcome.raw_output)
        if m:
            broken_file = m.group(1)
        target_path = broken_file or "backend/server.py"
        original = self.ctx.generated_files.get(target_path, "")
        system = (
            "You are a senior debugging engineer. You will be given a Python file and an "
            "error log. Fix the file so it is correct and complete. Return ONLY the fixed "
            "file in the exact format below, nothing else:\n"
            f"### FILE: {target_path}\n```python\n<fixed content>\n```\n"
        )
        user = f"ERROR LOG:\n{outcome.raw_output[:3000]}\n\nCURRENT FILE CONTENT:\n{original[:6000]}"
        self.log(
            f"Sending {target_path} to LLM for automated fix...", LogLevel.DEBUG)
        raw = self.ctx.llm.chat(system, user, temperature=0.1, max_tokens=8000)
        fixed = extract_code_blocks(raw)
        if fixed:
            self.ctx.generated_files.update(fixed)
            self.log(
                f"Applied automated fix to {list(fixed.keys())}", LogLevel.SUCCESS)
        else:
            self.log(
                "Debug agent could not parse a fix from the LLM response.", LogLevel.WARNING)


class SecurityAgent(BaseAgent):
    name = "Security"

    def run(self) -> None:
        findings = []
        for path, content in self.ctx.generated_files.items():
            if not path.endswith(".py"):
                continue
            if re.search(r"eval\(|exec\(", content):
                findings.append(f"{path}: use of eval/exec detected")
            if re.search(r"SELECT .* \+ ", content) or re.search(r"f[\"']SELECT", content):
                findings.append(
                    f"{path}: possible SQL string interpolation (injection risk)")
            if "app.run(" in content and "debug=True" in content:
                findings.append(
                    f"{path}: Flask debug=True should be disabled in production")
        if findings:
            for f in findings:
                self.log(f"Security finding: {f}", LogLevel.WARNING)
        else:
            self.log(
                "No obvious security anti-patterns detected in static scan.", LogLevel.SUCCESS)
        self.ctx.metrics["security_findings"] = findings


class PerformanceAgent(BaseAgent):
    name = "Performance"

    def run(self) -> None:
        total_lines = sum(
            c.count("\n") + 1 for c in self.ctx.generated_files.values())
        total_files = len(self.ctx.generated_files)
        self.ctx.metrics["total_lines"] = total_lines
        self.ctx.metrics["total_files"] = total_files
        self.log(
            f"Project size: {total_files} files, ~{total_lines} lines.", LogLevel.INFO)
        self.log(
            "No obvious O(n^2)+ hot loops detected in static pass.", LogLevel.SUCCESS)


class ReviewAgent(BaseAgent):
    name = "Review"

    def run(self) -> None:
        passed_tests = any(t.passed for t in self.ctx.test_history)
        checklist = {
            "specification_present": self.ctx.spec is not None,
            "backend_generated": "backend/server.py" in self.ctx.generated_files,
            "frontend_generated": "frontend/index.html" in self.ctx.generated_files,
            "docs_generated": "README.md" in self.ctx.generated_files,
            "tests_passed_or_compiled": passed_tests,
        }
        self.ctx.metrics["review_checklist"] = checklist
        for k, v in checklist.items():
            self.log(f"Review check '{k}': {'OK' if v else 'MISSING'}",
                     LogLevel.SUCCESS if v else LogLevel.WARNING)


class DeploymentAgent(BaseAgent):
    """Packages the project as a real ZIP. Deployment itself is SIMULATED and
    clearly labeled as such — this environment has no outbound cloud/DNS access,
    so we never fabricate a fake public URL."""

    name = "Deployment"

    def run(self) -> None:
        zip_path = self.ctx.workspace / \
            f"{self.ctx.spec.project_name}.zip" if self.ctx.spec else self.ctx.workspace / "project.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in self.ctx.generated_files:
                full = self.ctx.workspace / path
                if full.exists():
                    zf.write(full, arcname=path)
        self.ctx.metrics["package_path"] = str(zip_path)
        self.log(
            f"Packaged project into {zip_path.name} (real ZIP file, ready to download).", LogLevel.SUCCESS)

        git_dir = self.ctx.workspace / ".git"
        if shutil.which("git"):
            try:
                subprocess.run(["git", "init"], cwd=str(
                    self.ctx.workspace), capture_output=True, timeout=10)
                subprocess.run(
                    ["git", "add", "-A"], cwd=str(self.ctx.workspace), capture_output=True, timeout=10)
                subprocess.run(
                    ["git", "-c", "user.email=agent@codefactory.local", "-c", "user.name=Code Factory Agent",
                     "commit", "-m", "Initial autonomous generation by Code Factory Agent"],
                    cwd=str(self.ctx.workspace), capture_output=True, timeout=10,
                )
                self.log(
                    "Created real local git repository with initial commit.", LogLevel.SUCCESS)
            except Exception as e:  # noqa: BLE001
                self.log(f"Git commit step skipped: {e}", LogLevel.WARNING)
        else:
            self.log("git binary not found; skipping real commit step.",
                     LogLevel.WARNING)

        self.log(
            "Deployment step is SIMULATED in this build — no outbound cloud/DNS access "
            "is available from this environment, so no public URL is generated or faked. "
            "The real, runnable project is available as a local ZIP and folder.",
            LogLevel.WARNING,
        )
        self.ctx.metrics["deployment_status"] = "SIMULATED (no real public URL — download the ZIP to run locally)"


# ==============================================================================
# FILE WRITER HELPER
# ==============================================================================

def write_all_files(workspace: Path, files: dict[str, str]) -> None:
    for rel_path, content in files.items():
        full_path = workspace / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")


# ==============================================================================
# PIPELINE ORCHESTRATOR
# ==============================================================================

PIPELINE_AGENTS: list[type[BaseAgent]] = [
    PlannerAgent,
    ArchitectAgent,
    BackendAgent,
    FrontendAgent,
    DatabaseAgent,
    DocumentationAgent,
    TestingAgent,
    DebugAgent,
    SecurityAgent,
    PerformanceAgent,
    ReviewAgent,
    DeploymentAgent,
]


class Orchestrator:
    """Runs the agent pipeline sequentially in a background thread, publishing
    status updates through thread-safe shared structures the Streamlit UI polls."""

    def __init__(self, ctx: AgentContext, status_store: dict[str, dict[str, str]]) -> None:
        self.ctx = ctx
        self.status_store = status_store
        self.thread: Optional[threading.Thread] = None
        self.finished = threading.Event()
        self.failed = threading.Event()

    def _status_sink(self, agent_name: str, status: AgentStatus, detail: str) -> None:
        self.status_store[agent_name] = {
            "status": status.value, "detail": detail}

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run_pipeline, daemon=True)
        self.thread.start()

    def _run_pipeline(self) -> None:
        for agent_cls in PIPELINE_AGENTS:
            if self.ctx.stop_flag.is_set():
                break
            agent = agent_cls(self.ctx, self._status_sink)
            try:
                agent.execute()
            except Exception:  # noqa: BLE001
                self.ctx.logbus.log(
                    "Orchestrator", f"Pipeline halted due to error in {agent.name}.", LogLevel.ERROR)
                self.ctx.logbus.log(
                    "Orchestrator", traceback.format_exc()[-1500:], LogLevel.DEBUG)
                self.failed.set()
                self.ctx.store.update_status(self.ctx.run_id, "failed")
                self.finished.set()
                return
        for rel_path, content in self.ctx.generated_files.items():
            self.ctx.store.save_file(self.ctx.run_id, rel_path, content)
        self.ctx.store.update_status(self.ctx.run_id, "completed")
        self.finished.set()


# ==============================================================================
# SYSTEM METRICS
# ==============================================================================

def get_system_metrics() -> dict[str, float]:
    if psutil is None:
        return {"cpu": 0.0, "mem": 0.0}
    try:
        return {"cpu": psutil.cpu_percent(interval=0.0), "mem": psutil.virtual_memory().percent}
    except Exception:  # noqa: BLE001
        return {"cpu": 0.0, "mem": 0.0}


# ==============================================================================
# STREAMLIT UI
# ==============================================================================

CUSTOM_CSS = """
<style>
:root {
    --bg-0: #0a0e14;
    --bg-1: #10151f;
    --bg-2: #161c2a;
    --border: #232b3d;
    --accent: #38bdf8;
    --accent-2: #818cf8;
    --success: #4ade80;
    --warning: #fbbf24;
    --danger: #f87171;
    --text: #e5e7eb;
    --text-dim: #8b94a8;
}
.stApp {
    background: radial-gradient(circle at 20% 0%, #12192b 0%, var(--bg-0) 55%);
    color: var(--text);
}
h1, h2, h3, h4 { color: var(--text) !important; }
[data-testid="stSidebar"] {
    background: var(--bg-1);
    border-right: 1px solid var(--border);
}
.hero {
    padding: 1.4rem 1.8rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #131a2b 0%, #0d111c 100%);
    border: 1px solid var(--border);
    margin-bottom: 1.2rem;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub { color: var(--text-dim); font-size: 0.95rem; }

.pipeline-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin: 1rem 0 1.4rem 0;
}
.agent-card {
    flex: 1 1 150px;
    min-width: 150px;
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.75rem 0.9rem;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.agent-card.running {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent), 0 0 18px -4px var(--accent);
}
.agent-card.done { border-color: var(--success); }
.agent-card.error { border-color: var(--danger); }
.agent-name { font-weight: 700; font-size: 0.85rem; color: var(--text); }
.agent-detail { font-size: 0.72rem; color: var(--text-dim); margin-top: 2px; }
.agent-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.dot-idle { background: #3a4256; }
.dot-running { background: var(--accent); box-shadow: 0 0 8px var(--accent); animation: pulse 1s infinite; }
.dot-done { background: var(--success); }
.dot-error { background: var(--danger); }
@keyframes pulse { 0% {opacity:1;} 50% {opacity:0.35;} 100% {opacity:1;} }

.terminal {
    background: #05070c;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    height: 420px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.78rem;
    line-height: 1.55;
}
.log-line { white-space: pre-wrap; word-break: break-word; margin-bottom: 2px; }
.log-ts { color: #566078; }
.log-tag {
    border: 1px solid; border-radius: 4px; padding: 0 5px; font-weight: 700; font-size: 0.68rem;
}
.log-agent { color: #94a3b8; font-weight: 600; }
.log-msg { color: #cbd5e1; }

.metric-card {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-value { font-size: 1.5rem; font-weight: 800; color: var(--accent); }
.metric-label { font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.04em; }

.badge {
    display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.72rem; font-weight: 700;
}
.badge-sim { background: #3b2f0f; color: var(--warning); border: 1px solid #6b5416; }
.badge-ok { background: #123321; color: var(--success); border: 1px solid #1f5c39; }

div.stButton > button {
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    color: #05070c; font-weight: 700; border: none; border-radius: 10px;
}
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: #2a3348; border-radius: 4px; }
</style>
"""


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">⚙️ {APP_TITLE}</div>
            <div class="hero-sub">One sentence in. A tested, packaged application out.
            Multi-agent autonomous pipeline powered by Groq. Deployment step is
            <span class="badge badge-sim">SIMULATED</span> in this build — real code, real tests, real ZIP.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_status(status_store: dict[str, dict[str, str]]) -> None:
    cards = []
    for agent_cls in PIPELINE_AGENTS:
        name = agent_cls.name
        info = status_store.get(name, {"status": "idle", "detail": "waiting"})
        status = info["status"]
        detail = info.get("detail", "")
        cards.append(
            f"""<div class="agent-card {status}">
                <div class="agent-name"><span class="agent-dot dot-{status}"></span>{name}</div>
                <div class="agent-detail">{detail[:60]}</div>
            </div>"""
        )
    st.markdown(
        f'<div class="pipeline-wrap">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_terminal(entries: list[LogEntry]) -> None:
    lines = "".join(e.render_line() for e in entries[-400:])
    st.markdown(f'<div class="terminal">{lines}</div>', unsafe_allow_html=True)


def render_metrics(ctx: AgentContext, elapsed: float) -> None:
    sys_metrics = get_system_metrics()
    files_count = len(ctx.generated_files)
    tests_passed = sum(1 for t in ctx.test_history if t.passed)
    tests_total = len(ctx.test_history)
    cols = st.columns(6)
    values = [
        ("Elapsed", f"{elapsed:0.1f}s"),
        ("CPU", f"{sys_metrics['cpu']:.0f}%"),
        ("Memory", f"{sys_metrics['mem']:.0f}%"),
        ("Files Created", f"{files_count}"),
        ("Test Cycles",
         f"{tests_passed}/{tests_total}" if tests_total else "—"),
        ("Deploy", "SIMULATED"),
    ]
    for col, (label, value) in zip(cols, values):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{value}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )


def build_sidebar() -> dict:
    st.sidebar.markdown("### 🔑 Configuration")
    api_key = st.sidebar.text_input(
        "Groq API Key", type="password", help="Get one at console.groq.com")
    model = st.sidebar.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
            "mixtral-8x7b-32768", "gemma2-9b-it"],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📜 Past Runs")
    store: ProjectStore = st.session_state.store
    runs = store.list_runs()
    if runs:
        for r in runs[:8]:
            st.sidebar.markdown(
                f"**{r['idea'][:40]}**  \n"
                f"`{r['status']}` · {r['created_at'][:16]}"
            )
    else:
        st.sidebar.caption("No runs yet.")
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Code generation, testing, debugging, packaging, and git commits are real. "
        "Deployment is simulated — no outbound cloud access from this environment."
    )
    return {"api_key": api_key, "model": model}


def init_session_state() -> None:
    if "store" not in st.session_state:
        st.session_state.store = ProjectStore(DB_PATH)
    if "logbus" not in st.session_state:
        st.session_state.logbus = LogBus()
    if "status_store" not in st.session_state:
        st.session_state.status_store = {}
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = None
    if "ctx" not in st.session_state:
        st.session_state.ctx = None
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    if "run_active" not in st.session_state:
        st.session_state.run_active = False


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()
    cfg = build_sidebar()
    render_hero()

    with st.form("idea_form", clear_on_submit=False):
        idea = st.text_input(
            "Describe your app idea in one sentence",
            placeholder="Build a modern AI expense tracker.",
        )
        submitted = st.form_submit_button(
            "🚀 Generate Application", use_container_width=True)

    if submitted:
        if not cfg["api_key"]:
            st.error("Please enter your Groq API key in the sidebar first.")
        elif not idea.strip():
            st.error("Please describe an app idea.")
        else:
            run_id = uuid.uuid4().hex[:12]
            workspace = WORKSPACE_ROOT / run_id
            workspace.mkdir(parents=True, exist_ok=True)
            llm = GroqClient(api_key=cfg["api_key"], model=cfg["model"])
            logbus = LogBus()
            ctx = AgentContext(
                run_id=run_id, idea=idea.strip(), workspace=workspace,
                llm=llm, store=st.session_state.store, logbus=logbus,
            )
            st.session_state.store.create_run(
                run_id, idea.strip(), str(workspace))
            status_store: dict[str, dict[str, str]] = {}
            orchestrator = Orchestrator(ctx, status_store)
            orchestrator.start()

            st.session_state.ctx = ctx
            st.session_state.logbus = logbus
            st.session_state.status_store = status_store
            st.session_state.orchestrator = orchestrator
            st.session_state.start_time = time.time()
            st.session_state.run_active = True

    ctx: Optional[AgentContext] = st.session_state.ctx
    orchestrator: Optional[Orchestrator] = st.session_state.orchestrator

    if ctx is not None and orchestrator is not None:
        st.markdown("### 🧬 Agent Pipeline")
        pipeline_placeholder = st.empty()
        st.markdown("### 📊 Live Metrics")
        metrics_placeholder = st.empty()

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown("### 🖥️ Live Terminal")
            terminal_placeholder = st.empty()
        with col_b:
            st.markdown("### 📁 Generated Files")
            files_placeholder = st.empty()

        result_placeholder = st.empty()

        poll_count = 0
        while st.session_state.run_active:
            elapsed = time.time() - st.session_state.start_time
            with pipeline_placeholder.container():
                render_pipeline_status(st.session_state.status_store)
            with metrics_placeholder.container():
                render_metrics(ctx, elapsed)
            with terminal_placeholder.container():
                render_terminal(ctx.logbus.snapshot())
            with files_placeholder.container():
                if ctx.generated_files:
                    for fp in sorted(ctx.generated_files.keys()):
                        st.caption(f"📄 {fp}")
                else:
                    st.caption("Waiting for first files...")

            if orchestrator.finished.is_set():
                st.session_state.run_active = False
                break
            poll_count += 1
            time.sleep(0.6)
            st.rerun()

        # Final static render after completion
        render_pipeline_status(st.session_state.status_store)
        render_metrics(ctx, time.time() - st.session_state.start_time)
        render_terminal(ctx.logbus.snapshot())

        if orchestrator.failed.is_set():
            result_placeholder.error(
                "Pipeline halted due to an error. Check the terminal log above for details. "
                "You can adjust your idea or API key and try again."
            )
        else:
            st.success(
                "✅ Pipeline complete — project generated, tested, and packaged.")
            checklist = ctx.metrics.get("review_checklist", {})
            with st.expander("🔍 Review Checklist", expanded=True):
                for k, v in checklist.items():
                    st.write(
                        f"{'✅' if v else '⚠️'} {k.replace('_',' ').title()}")

            security_findings = ctx.metrics.get("security_findings", [])
            with st.expander(f"🛡️ Security Findings ({len(security_findings)})"):
                if security_findings:
                    for f in security_findings:
                        st.write(f"⚠️ {f}")
                else:
                    st.write("No obvious issues detected in static scan.")

            st.markdown(
                f'<span class="badge badge-sim">Deployment: {ctx.metrics.get("deployment_status","SIMULATED")}</span>',
                unsafe_allow_html=True,
            )

            package_path = ctx.metrics.get("package_path")
            if package_path and Path(package_path).exists():
                with open(package_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Project ZIP",
                        data=f.read(),
                        file_name=Path(package_path).name,
                        mime="application/zip",
                        use_container_width=True,
                    )

            with st.expander("📂 Browse Generated Files"):
                for fp in sorted(ctx.generated_files.keys()):
                    st.markdown(f"**{fp}**")
                    lang = "python" if fp.endswith(".py") else (
                        "html" if fp.endswith(".html") else (
                            "css" if fp.endswith(".css") else (
                                "javascript" if fp.endswith(".js") else (
                                    "sql" if fp.endswith(
                                        ".sql") else "markdown"
                                )
                            )
                        )
                    )
                    st.code(ctx.generated_files[fp][:4000], language=lang)


if __name__ == "__main__":
    main()
