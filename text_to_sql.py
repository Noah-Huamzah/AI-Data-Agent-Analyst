import os, json, time, uuid, pickle, hashlib, logging, asyncio
from typing import Optional, List, Tuple, Dict, Any
#uvicorn app:app --reload
import httpx
import numpy as np
import requests
import oracledb
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config & Logging
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL",     "http://localhost:11434")
GENERATION_MODEL  = os.getenv("GENERATION_MODEL",    "mistral")#"qwen2.5-coder:3b","codellama"
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL",     "nomic-embed-text")
DB_USER           = os.getenv("DB_USER",              "system")
DB_PASSWORD       = os.getenv("DB_PASSWORD")
DB_DSN            = os.getenv("DB_DSN",               "localhost/FREEXDB")
SCHEMA_CACHE_FILE = os.getenv("SCHEMA_CACHE_FILE",   "schema_embeddings.pkl")
TOP_K_SCHEMA      = int(os.getenv("TOP_K_SCHEMA",    "3"))
MAX_RETRIES       = int(os.getenv("MAX_RETRIES",      "2"))
OLLAMA_TIMEOUT    = int(os.getenv("OLLAMA_TIMEOUT",   "90"))
PIPELINE_TIMEOUT  = int(os.getenv("PIPELINE_TIMEOUT", "180"))   # asyncio.wait_for cap (seconds)
MAX_ROWS          = int(os.getenv("MAX_ROWS",          "500"))  # DB-level hard cap
UI_MAX_ROWS       = int(os.getenv("UI_MAX_ROWS",       "20"))   # rows returned in API response
CACHE_TTL_HOURS   = int(os.getenv("CACHE_TTL_HOURS",  "24"))
OLLAMA_FORCE_JSON = os.getenv("OLLAMA_FORCE_JSON",    "false").lower() == "true"
CONFIDENCE_THRESH = float(os.getenv("CONFIDENCE_THRESH", "0.5"))
REDIS_URL         = os.getenv("REDIS_URL", "")
CONV_TURNS        = int(os.getenv("CONVERSATION_TURNS", "5"))

ALLOWED_TABLES = [
    "CUSTOMERS", "ORDERS", "ORDER_ITEMS",
    "PRODUCTS", "CATEGORIES", "PAYMENTS", "SHIPMENTS"
]

# Used by both is_safe_sql() and validate_output()
BLOCKED_KEYWORDS = {
    "drop", "delete", "truncate", "insert",
    "update", "alter", "create", "grant", "exec", "execute"
}

# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = """
### EXAMPLE
Q: Top 3 cities by revenue
A: {"query": "SELECT c.city, SUM(oi.quantity * p.price) AS revenue FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id GROUP BY c.city ORDER BY revenue DESC FETCH FIRST 3 ROWS ONLY", "explanation": "Groups customers by city and calculates revenue via order items and products.", "answer": "Top cities generate the most revenue.", "confidence": 0.96, "assumptions": "none"}
"""
_async_client: Optional[httpx.AsyncClient] = None

def get_async_client() -> httpx.AsyncClient:
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(
            timeout=OLLAMA_TIMEOUT,
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20
            )
        )
    return _async_client
# ---------------------------------------------------------------------------
# Standardised response builders
# ---------------------------------------------------------------------------

def make_error(code: str, message: str, detail: str = "", query_id: str = "") -> dict:
    """Every error from the pipeline has the same shape."""
    return {
        "success":       False,
        "query_id":      query_id or str(uuid.uuid4()),
        "error_code":    code,
        "error_message": message,
        "detail":        detail,
        # Standardised fields present but empty so callers don't KeyError
        "sql":           "",
        "answer":        "",
        "confidence":    0.0,
    }


def make_success(query_id: str, question: str, sql: str, explanation: str,
                 assumptions: str, confidence: float, data: dict,
                 answer: str, latency_ms: int, cached: bool) -> dict:
    """Standardised success response — shape never changes."""
    return {
        "success":         True,
        "query_id":        query_id,
        # Top-level summary fields (what the API surface exposes)
        "answer":          answer,
        "sql":             sql,
        "confidence":      confidence,
        # Detail fields
        "question":        question,
        "sql_explanation": explanation,
        "assumptions":     assumptions,
        "data": {
            "columns":   data["columns"],
            "rows":      data["rows"][:UI_MAX_ROWS],   # cap for UI
            "row_count": data["row_count"],             # real total
            "truncated": data["row_count"] > UI_MAX_ROWS,
        },
        "latency_ms":      latency_ms,
        "cached":          cached,
    }

# ---------------------------------------------------------------------------
# Safety gate  (dedicated, separate from structural validation)
# ---------------------------------------------------------------------------

def is_safe_sql(sql: str) -> bool:
    """
    Hard safety check — runs AFTER the LLM generates SQL and BEFORE DB execution.
    Returns False if the query is not a plain SELECT or contains blocked keywords.
    """
    sql_lower = sql.strip().lower()
    if not sql_lower.startswith("select"):
        return False
    # Token-boundary check to avoid matching e.g. column name "updated_at"
    tokens = set(sql_lower.replace("(", " ").replace(")", " ").split())
    return not bool(tokens & BLOCKED_KEYWORDS)

# ---------------------------------------------------------------------------
# DB Connection Pool
# ---------------------------------------------------------------------------

_pool: Optional[oracledb.ConnectionPool] = None

def get_pool() -> oracledb.ConnectionPool:
    global _pool
    if _pool is None:
        if not DB_PASSWORD:
            raise EnvironmentError("DB_PASSWORD env var is not set.")
        _pool = oracledb.create_pool(
            user=DB_USER, password=DB_PASSWORD, dsn=DB_DSN,
            min=1, max=5, increment=1
        )
        log.info("Oracle connection pool created.")
    return _pool

# ---------------------------------------------------------------------------
# Query History
# ---------------------------------------------------------------------------

def ensure_history_table() -> None:
    ddl = """
        CREATE TABLE query_history (
            id          NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            query_id    VARCHAR2(36),
            question    VARCHAR2(1000),
            sql_text    CLOB,
            success     NUMBER(1),
            error_msg   VARCHAR2(4000),
            row_count   NUMBER,
            latency_ms  NUMBER,
            confidence  NUMBER(5,2),
            cached      NUMBER(1) DEFAULT 0,
            created_at  TIMESTAMP DEFAULT SYSTIMESTAMP
        )
    """
    try:
        with get_pool().acquire() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()
        log.info("query_history table created.")
    except oracledb.DatabaseError as e:
        if "ORA-00955" not in str(e):
            log.warning(f"Could not create query_history: {e}")


def log_query_history(
    query_id: str, question: str, sql: str, success: bool,
    row_count: int = 0, latency_ms: int = 0,
    confidence: float = 0.0, cached: bool = False, error_msg: str = ""
) -> None:
    try:
        with get_pool().acquire() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO query_history
                        (query_id, question, sql_text, success, error_msg,
                         row_count, latency_ms, confidence, cached)
                    VALUES (:1,:2,:3,:4,:5,:6,:7,:8,:9)
                """, (query_id, question[:1000], sql, int(success), error_msg[:4000],
                      row_count, latency_ms, round(confidence, 2), int(cached)))
            conn.commit()
    except Exception as e:
        log.warning(f"History log failed: {e}")


def get_history(limit: int = 20) -> List[Dict]:
    try:
        with get_pool().acquire() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT query_id, question, sql_text, success, row_count,
                           latency_ms, confidence, cached, created_at
                    FROM   query_history
                    ORDER  BY created_at DESC
                    FETCH  FIRST :1 ROWS ONLY
                """, (limit,))
                cols = [c[0].lower() for c in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as e:
        log.warning(f"Could not fetch history: {e}")
        return []

# ---------------------------------------------------------------------------
# Result Cache  (in-memory with optional Redis backend)
# ---------------------------------------------------------------------------

_mem_cache: Dict[str, dict] = {}

def _cache_key(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

def cache_get(question: str) -> Optional[dict]:
    key = _cache_key(question)
    if REDIS_URL:
        try:
            import redis
            raw = redis.from_url(REDIS_URL).get(key)
            if raw:
                log.info("Redis cache hit.")
                return json.loads(raw)
        except Exception as e:
            log.warning(f"Redis get failed: {e}")
    return _mem_cache.get(key)

def cache_set(question: str, result: dict) -> None:
    key = _cache_key(question)
    # Don't cache query_id — each cache hit should produce a fresh one
    storable = {k: v for k, v in result.items() if k != "query_id"}
    if REDIS_URL:
        try:
            import redis
            redis.from_url(REDIS_URL).setex(key, 3600, json.dumps(storable, default=str))
        except Exception as e:
            log.warning(f"Redis set failed: {e}")
    _mem_cache[key] = storable

def cache_clear() -> int:
    count = len(_mem_cache)
    _mem_cache.clear()
    log.info(f"Cache cleared ({count} entries).")
    return count

# ---------------------------------------------------------------------------
# SQL Execution
# ---------------------------------------------------------------------------

def enforce_row_limit(sql: str) -> str:
    """Inject a hard DB-level cap if no FETCH FIRST clause exists."""
    if "fetch first" not in sql.lower():
        return f"{sql.rstrip().rstrip(';')} FETCH FIRST {MAX_ROWS} ROWS ONLY"
    return sql

def run_sql(query: str) -> dict:
    query = enforce_row_limit(query)
    try:
        with get_pool().acquire() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                columns = [col[0] for col in cursor.description]
                rows    = cursor.fetchall()
                return {"columns": columns, "rows": rows, "row_count": len(rows)}
    except oracledb.DatabaseError as e:
        error_obj, = e.args
        return {"error": error_obj.message}
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------------------------
# Embedding Cache  (avoids re-calling Ollama for repeated identical strings)
# ---------------------------------------------------------------------------

_embedding_cache: Dict[str, list] = {}

def get_embedding(text: str) -> list:
    """Raw embedding call — no cache."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["embedding"]

def get_embedding_cached(text: str) -> list:
    """Cached embedding — skips Ollama if we've seen this string before."""
    if text in _embedding_cache:
        log.debug(f"Embedding cache hit for: {text[:60]}")
        return _embedding_cache[text]
    emb = get_embedding(text)
    _embedding_cache[text] = emb
    return emb

def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

async def get_embedding_cached_async(text: str) -> list:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_embedding_cached, text)

async def get_relevant_schema_async(query: str, top_k: int = TOP_K_SCHEMA) -> str:
    global _schema_cache

    if _schema_cache is None or is_cache_stale():
        # run blocking load in thread
        loop = asyncio.get_event_loop()
        _schema_cache = await loop.run_in_executor(None, load_or_build_schema_cache)

    # run embedding in parallel
    query_vec = await get_embedding_cached_async(query)

    scores = [
        cosine_similarity(query_vec, e)
        for e in _schema_cache["embeddings"]
    ]

    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return "\n".join([_schema_cache["docs"][i] for i in top_idx])

# ---------------------------------------------------------------------------
# Schema Cache with TTL auto-refresh
# ---------------------------------------------------------------------------

def is_cache_stale() -> bool:
    if not os.path.exists(SCHEMA_CACHE_FILE):
        return True
    age_hours = (time.time() - os.path.getmtime(SCHEMA_CACHE_FILE)) / 3600
    return age_hours > CACHE_TTL_HOURS
def normalize_row(row):
    new_row = []
    for val in row:
        try:
            if hasattr(val, "read"):  # LOB (CLOB/BLOB)
                val = val.read()
        except:
            pass
        new_row.append(val)
    return new_row


def get_schema() -> dict:
    table_list = ",".join([f"'{t}'" for t in ALLOWED_TABLES])
    with get_pool().acquire() as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT table_name, column_name, data_type
                FROM   all_tab_columns
                WHERE  owner = '{DB_USER.upper()}'
                  AND  table_name IN ({table_list})
                ORDER  BY table_name, column_id
            """)
            rows = cursor.fetchall()
            rows = [normalize_row(r) for r in rows]
            rows = rows[:20]
    schema: dict = {}
    for table, column, dtype in rows:
        schema.setdefault(table, []).append(f"{column}({dtype})")
    return schema

def build_schema_docs(schema: dict) -> List[str]:
    return [f"{table}: {', '.join(cols)}" for table, cols in schema.items()]

def load_or_build_schema_cache(force: bool = False) -> dict:
    if not force and not is_cache_stale():
        with open(SCHEMA_CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        log.info("Loaded schema embeddings from disk cache.")
        return cache
    reason = "forced" if force else "stale/missing"
    log.info(f"Building schema embeddings ({reason})…")
    schema     = get_schema()
    docs       = build_schema_docs(schema)
    # Use cached embeddings here too — schema strings rarely change
    embeddings = [get_embedding_cached(doc) for doc in docs]
    cache      = {"docs": docs, "embeddings": embeddings, "built_at": time.time()}
    with open(SCHEMA_CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    log.info("Schema embeddings saved to disk.")
    return cache

_schema_cache: Optional[Dict] = None


def get_relevant_schema(query: str, top_k: int = TOP_K_SCHEMA) -> str:
    global _schema_cache
    if _schema_cache is None or is_cache_stale():
        _schema_cache = load_or_build_schema_cache()
    # Use cached embedding for the user query too
    query_vec = get_embedding_cached(query)
    scores    = [cosine_similarity(query_vec, e) for e in _schema_cache["embeddings"]]
    top_idx   = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return "\n".join([_schema_cache["docs"][i] for i in top_idx])

# ---------------------------------------------------------------------------
# Table Relationships
# ---------------------------------------------------------------------------

def get_table_relationships() -> List[Dict]:
    with get_pool().acquire() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT a.table_name, a.column_name,
                       c_pk.table_name AS ref_table, b.column_name AS ref_column
                FROM   all_cons_columns a
                       JOIN all_constraints c
                         ON a.owner = c.owner AND a.constraint_name = c.constraint_name
                       JOIN all_constraints c_pk
                         ON c.r_owner = c_pk.owner AND c.r_constraint_name = c_pk.constraint_name
                       JOIN all_cons_columns b
                         ON c_pk.owner = b.owner AND c_pk.constraint_name = b.constraint_name
                            AND a.position = b.position
                WHERE  c.constraint_type = 'R' AND a.owner = :owner
            """, owner=DB_USER.upper())
            rows = cursor.fetchall()
            rows = [normalize_row(r) for r in rows]
            rows = rows[:20]
    return [{"table": r[0], "column": r[1], "ref_table": r[2], "ref_column": r[3]} for r in rows]

def format_relationships(rels: List[Dict]) -> str:
    return "\n".join(
        f"{r['table']}.{r['column']} -> {r['ref_table']}.{r['ref_column']}"
        for r in rels
    )
    
def get_cached_relationships() -> str:
    global _relationship_cache

    if _relationship_cache is not None:
        return _relationship_cache

    rels = get_table_relationships()
    formatted = format_relationships(rels[:8])

    _relationship_cache = formatted
    log.info("Cached table relationships.")

    return formatted

_relationship_cache: Optional[str] = None
# ---------------------------------------------------------------------------
# JSON Utilities
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON in LLM response: {text[:200]}")
    raw = text[start:end].replace("```json","").replace("```","").strip()
    return json.loads(raw)

# ---------------------------------------------------------------------------
# Structural Validation  (LLM output shape + confidence check)
# ---------------------------------------------------------------------------

def validate_output(data: dict) -> Tuple[bool, str]:
    """Check that the LLM returned a well-formed dict with required keys and acceptable confidence."""
    if not isinstance(data, dict):
        return False, "Not a dict"
    if "query" not in data or "explanation" not in data:
        return False, "Missing query/explanation keys"
    confidence = float(data.get("confidence", 1.0))
    if confidence < CONFIDENCE_THRESH:
        return False, f"Low confidence ({confidence:.2f}): {data.get('assumptions','')}"
    # Note: keyword safety is handled separately by is_safe_sql() after this check
    return True, ""

def should_retry_sql(error_msg: str) -> bool:
    """Decide whether an LLM fix is worth trying."""
    if not error_msg:
        return False

    err = error_msg.lower()

    retryable = [
        "ora-00904",  # invalid column
        "ora-01722",  # invalid number
        "ora-01843",  # invalid date
        "ora-00936",  # missing expression
        "ora-00933",  # sql not properly ended
    ]

    non_retryable = [
        "ora-00942",  # table/view does not exist
        "ora-01031",  # insufficient privileges
        "timeout",
    ]

    if any(e in err for e in non_retryable):
        return False

    return any(e in err for e in retryable)
# ---------------------------------------------------------------------------
# Async Ollama wrappers
# ---------------------------------------------------------------------------

async def _call_ollama_async(prompt: str, expect_json: bool = True) -> str:
    payload: Dict[str, Any] = {
        "model": GENERATION_MODEL, "prompt": prompt, "stream": False
    }
    if expect_json and OLLAMA_FORCE_JSON:
        payload["format"] = "json"
    try:
        client = get_async_client()
        resp = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        if resp.status_code != 200:
            log.error(f"Ollama {resp.status_code}: {resp.text[:400]}")
        resp.raise_for_status()
    except httpx.ConnectError:
        raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_BASE_URL}. Try: ollama serve")
    result = resp.json()
    if "response" not in result:
        raise RuntimeError(f"No 'response' in Ollama output: {result}")
    return result["response"]

def _call_ollama(prompt: str, expect_json: bool = True) -> str:
    payload: Dict[str, Any] = {
        "model": GENERATION_MODEL, "prompt": prompt, "stream": False
    }
    if expect_json and OLLAMA_FORCE_JSON:
        payload["format"] = "json"
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT
        )
        if resp.status_code != 200:
            log.error(f"Ollama {resp.status_code}: {resp.text[:400]}")
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_BASE_URL}. Try: ollama serve")
    result = resp.json()
    if "response" not in result:
        raise RuntimeError(f"No 'response' in Ollama output: {result}")
    return result["response"]

# ---------------------------------------------------------------------------
# Model Warm-up  (called ONCE at startup — NOT per request)
# ---------------------------------------------------------------------------

async def warmup_model_async() -> None:
    """
    Pre-load the model into Ollama's RAM.
    Call this once at application startup (e.g. FastAPI lifespan).
    Do NOT call inside query_pipeline_async — that would run it on every request.
    """
    log.info(f"Warming up '{GENERATION_MODEL}'…")
    try:
        await _call_ollama_async("Reply OK.", expect_json=False)
        log.info("Warm-up complete.")
    except Exception as e:
        log.warning(f"Warm-up failed (non-fatal): {e}")

def warmup_model() -> None:
    """Sync version — used by CLI __main__."""
    log.info(f"Warming up '{GENERATION_MODEL}'…")
    try:
        _call_ollama("Reply OK.", expect_json=False)
        log.info("Warm-up complete.")
    except Exception as e:
        log.warning(f"Warm-up failed (non-fatal): {e}")

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_sql_prompt(user_query: str, schema_context: str,
                      relationships: str, history_context: str = "") -> str:
    history_block = (
        f"\n### CONVERSATION HISTORY (use for follow-up questions)\n{history_context}\n"
        if history_context else ""
    )
    return f"""
### ROLE
You are an expert Oracle SQL data engineer. Return ONLY valid JSON, nothing else.

### DATABASE SCHEMA
{schema_context}

### TABLE RELATIONSHIPS
{relationships}
{history_block}
{FEW_SHOT_EXAMPLES}

### RULES
- Use ONLY tables and columns from the schema.
- Never invent table or column names.
- Use Oracle SQL syntax (FETCH FIRST N ROWS ONLY, not LIMIT).
- Revenue = ORDER_ITEMS.quantity * PRODUCTS.price — always JOIN PRODUCTS for revenue.
- Use proper JOIN conditions from the relationships above.
- Output ONLY the JSON object, no markdown, no preamble.
- "answer" must be a short business insight (1 sentence, non-technical).

### OUTPUT FORMAT
{{"query": "<valid Oracle SQL>", "explanation": "<one-line summary>","answer": "<1-line business insight>", "confidence": <0.0-1.0>, "assumptions": "<or 'none'>"}}

### USER REQUEST
{user_query}
"""

# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

async def ask_llm_with_context_async(user_query: str, history_context: str = "") -> dict:
    schema_context = await get_relevant_schema_async(user_query)
    relationships = get_cached_relationships()
    prompt        = _build_sql_prompt(user_query, schema_context, relationships, history_context)
    raw           = await _call_ollama_async(prompt, expect_json=True)
    log.debug(f"Raw LLM: {raw}")
    return extract_json(raw)

async def fix_sql_async(user_input: str, bad_query: str, error_msg: str) -> dict:
    prompt = f"""
### ROLE
You are an expert Oracle SQL debugger. Return ONLY valid JSON.

### USER REQUEST
{user_input}

### BROKEN QUERY
{bad_query}

### ORACLE ERROR
{error_msg}

### TASK
Fix the query. Preserve original intent, filters, aggregations, and limits.

### OUTPUT FORMAT
{{"query": "<fixed SQL>", "explanation": "<what changed>", "confidence": <0.0-1.0>, "assumptions": "<or 'none'>"}}
"""
    t0 = time.time()
    raw = await _call_ollama_async(prompt, expect_json=True)
    log.info(f"LLM call took {time.time() - t0:.2f}s")
    return extract_json(raw)


# ---------------------------------------------------------------------------
# Retry loop
# ---------------------------------------------------------------------------

async def generate_and_run_async(
    user_input:      str,
    history_context: str = "",
    retries:         int = MAX_RETRIES,
    query_id:        str = ""
) -> Tuple[dict, Optional[dict]]:
    """
    1. Generate SQL
    2. Structural validation (validate_output)
    3. Safety gate      (is_safe_sql)   ← NEW dedicated check
    4. Execute against DB
    5. On DB error → fix_sql and retry
    """
    sql_result: dict          = {}
    db_result: Optional[dict] = None

    for attempt in range(1, retries + 1):
        log.info(f"Attempt {attempt}/{retries}…")

        # --- Generate ---
        try:
            if attempt == 1 or "query" not in sql_result:
                sql_result = await ask_llm_with_context_async(user_input, history_context)
        except Exception as e:
            log.warning(f"LLM failed: {e}")
            continue

        # --- Structural validation ---
        valid, reason = validate_output(sql_result)
        if not valid:
            log.warning(f"Validation failed: {reason}")
            sql_result = {}
            continue

        # --- Safety gate (dedicated is_safe_sql check) ---
        if not is_safe_sql(sql_result["query"]):
            log.error(f"UNSAFE SQL blocked: {sql_result['query'][:120]}")
            return make_error("UNSAFE_SQL", "Query blocked for safety.", query_id=query_id), None

        log.info(f"SQL (conf={sql_result.get('confidence','?')}):\n{sql_result['query']}")

        # --- Execute ---
        db_result = run_sql(sql_result["query"])

        if "error" not in db_result:
            log.info(f"Success — {db_result['row_count']} rows.")
            return sql_result, db_result

        # --- Fix on DB error ---
        error_msg = db_result["error"]
        log.warning(f"DB error: {error_msg}")

        if not should_retry_sql(error_msg):
            log.warning("Skipping fix_sql (non-retryable error).")
            return make_error(
                "SQL_ERROR",
                "Query failed and is not retryable.",
                detail=error_msg,
                query_id=query_id
            ), None

        try:
            sql_result = await fix_sql_async(user_input, sql_result["query"], error_msg)
        except Exception as e:
            log.warning(f"fix_sql failed: {e}")
            sql_result = {}

    return make_error("MAX_RETRIES", f"Failed after {retries} attempts", query_id=query_id), None

# ---------------------------------------------------------------------------
# Conversation Session
# ---------------------------------------------------------------------------

class ConversationSession:
    def __init__(self):
        self.turns: List[Dict] = []

    def add(self, question: str, sql: str, insight: str) -> None:
        self.turns.append({"question": question, "sql": sql, "insight": insight})
        if len(self.turns) > CONV_TURNS:
            self.turns.pop(0)

    def as_context(self) -> str:
        if not self.turns:
            return ""
        return "\n".join(
            f"Turn {i}: Q={t['question']} | SQL={t['sql']}"
            for i, t in enumerate(self.turns, 1)
        )

    def clear(self) -> None:
        self.turns.clear()

    def to_list(self) -> List[Dict]:
        return list(self.turns)

# ---------------------------------------------------------------------------
# Main pipeline (async)
# ---------------------------------------------------------------------------

_default_session = ConversationSession()
    
async def _pipeline_core(
    user_input: str,
    session:    ConversationSession,
    query_id:   str
) -> dict:
    """Inner pipeline logic — wrapped by query_pipeline_async with a timeout guard."""

    # 1. Cache check
    cached = cache_get(user_input)
    if cached:
        log.info("Cache hit.")
        fresh = {**cached, "query_id": query_id, "cached": True}
        log_query_history(
            query_id, user_input, cached.get("sql",""), True,
            cached.get("data",{}).get("row_count",0), 0,
            cached.get("confidence",0), cached=True
        )
        return fresh

    t_start = time.time()

    # 2. Generate + Execute (retry loop)
    sql_result, db_result = await generate_and_run_async(
        user_input, session.as_context(), query_id=query_id
    )
    latency_ms = int((time.time() - t_start) * 1000)

    # Error path
    if not sql_result.get("success", True) or db_result is None:
        err = {**sql_result, "query_id": query_id, "question": user_input}
        log_query_history(
            query_id, user_input, "", False,
            error_msg=sql_result.get("error_message",""), latency_ms=latency_ms
        )
        return err

    # 3. Explain
    # Insight (no extra LLM call)
    if db_result["row_count"] == 0:
        insight = "No data found matching your query."
    else:
        insight = sql_result.get("answer", "(No insight provided)")

    confidence = float(sql_result.get("confidence", 0.0))

    result = make_success(
        query_id    = query_id,
        question    = user_input,
        sql         = sql_result["query"],
        explanation = sql_result.get("explanation",""),
        assumptions = sql_result.get("assumptions","none"),
        answer      = insight,   # ✅ NEW (standardised)
        confidence  = confidence,
        data        = db_result,
        latency_ms  = latency_ms,
        cached      = False,
    )

    # 4. Cache & log
    cache_set(user_input, result)
    log_query_history(
        query_id, user_input, sql_result["query"], True,
        db_result["row_count"], latency_ms, confidence
    )
    session.add(user_input, sql_result["query"], insight)
    return result


async def query_pipeline_async(
    user_input: str,
    session:    Optional[ConversationSession] = None
) -> dict:
    """
    Public async entry-point.
    Wraps _pipeline_core with asyncio.wait_for for a hard timeout cap.
    Warm-up is NOT called here — call warmup_model_async() once at startup.
    """
    query_id = str(uuid.uuid4())
    session  = session or _default_session
    log.info(f"\n{'='*60}\n[{query_id}] {user_input!r}\n{'='*60}")

    try:
        return await asyncio.wait_for(
            _pipeline_core(user_input, session, query_id),
            timeout=PIPELINE_TIMEOUT
        )
    except asyncio.TimeoutError:
        log.error(f"[{query_id}] Pipeline timed out after {PIPELINE_TIMEOUT}s")
        return make_error(
            "TIMEOUT",
            f"Request timed out after {PIPELINE_TIMEOUT} seconds. Try a simpler question.",
            query_id=query_id
        )


def query_pipeline(
    user_input: str,
    session:    Optional[ConversationSession] = None
) -> dict:
    """Sync wrapper — used by CLI."""
    return asyncio.get_event_loop().run_until_complete(
        query_pipeline_async(user_input, session)
    )

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_result(result: dict) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"🔑  Query ID  : {result.get('query_id','')}")
    print(f"❓  Question  : {result.get('question','')}")
    if not result.get("success"):
        print(f"❌  [{result.get('error_code','ERROR')}] {result.get('error_message','')}")
        if result.get("detail"):
            print(f"    {result['detail']}")
        return
    cached_tag = " (cached ⚡)" if result.get("cached") else ""
    print(f"✅  SQL{cached_tag}:\n    {result['sql']}")
    print(f"💡  {result.get('sql_explanation','')}")
    conf = result.get("confidence", 0)
    data = result.get("data", {})
    trunc = " (truncated)" if data.get("truncated") else ""
    print(f"🎯  Confidence: {conf:.0%}  |  Rows: {data.get('row_count',0)}{trunc}  |  "
          f"Latency: {result.get('latency_ms',0)} ms")
    print(f"📋  {data.get('columns','')}")
    for row in data.get("rows", [])[:5]:
        print(f"    {row}")
    if data.get("row_count",0) > 5:
        print(f"    … ({data['row_count'] - 5} more rows)")
    print(f"\n🔍  {result.get('answer','')}")
    print(sep)


if __name__ == "__main__":
    import sys
    ensure_history_table()
    warmup_model()                    # once, at CLI startup
    session = ConversationSession()
    while True:
        q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("\n💬 Ask: ").strip()
        if q.lower() in ("exit", "quit", ""):
            break
        print_result(query_pipeline(q, session))
        if len(sys.argv) > 1:
            break
