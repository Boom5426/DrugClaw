"""
TarKG - Drug Target Discovery Knowledge Graph Query Tool
Category: Drug-centric | Type: KG | Subcategory: Drug-Target Interaction (DTI)
Link: https://tarkg.ddtmlab.org/index
Paper: https://academic.oup.com/bioinformatics/article/40/10/btae598/7818343

TarKG integrates multi-source biomedical data (incl. TCM) with 171 relation types,
~100K nodes, ~1M edges for drug target discovery and repurposing.

Requires: pandas (pip install pandas)
Data: Pre-downloaded CSV files from https://tarkg.ddtmlab.org/download
"""

import os
import csv
import json
import sqlite3
import glob

# ── Default data path (change to your local path) ──────────────────────────
DATA_DIR = os.environ.get(
    "TARKG_DATA",
    "/blue/qsong1/wang.qing/AgentLLM/Survey100/resources_metadata/dti/TarKG",
)
DB_PATH = os.path.join(DATA_DIR, ".tarkg_index.db")


# ── 1. Build SQLite index (one-time) ───────────────────────────────────────
def _detect_sep(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
    return "\t" if "\t" in header else ","


def _load_csv(path: str) -> list[dict]:
    sep = _detect_sep(path)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=sep)
        return list(reader)


def _q(name: str) -> str:
    """Quote a column/table name to avoid SQLite reserved-word conflicts."""
    return f'"{name}"'


def build_index(data_dir: str = DATA_DIR, force: bool = False) -> str:
    """Build SQLite index from TarKG CSVs. Returns DB path."""
    db = os.path.join(data_dir, ".tarkg_index.db")
    if os.path.exists(db) and not force:
        return db

    conn = sqlite3.connect(db)
    c = conn.cursor()

    # ── nodes table ────────────────────────────────────────────────────
    nodes_file = os.path.join(data_dir, "TarKG_nodes.csv")
    rows = _load_csv(nodes_file)
    cols = list(rows[0].keys())

    # Smart column mapping: try known names, fallback to positional
    def _pick(candidates, columns, default_idx=0):
        for c in candidates:
            if c in columns:
                return c
        return columns[default_idx]

    id_col   = _pick(["unify_id", "node_id", "id"], cols, 0)
    name_col = _pick(["name", "node_name"], cols, 1)
    type_col = _pick(["kind", "node_type", "type"], cols, 2)
    print(f"  nodes mapping: id={id_col}, name={name_col}, type={type_col}")

    c.execute("DROP TABLE IF EXISTS nodes")
    c.execute(
        "CREATE TABLE nodes (node_id TEXT PRIMARY KEY, node_name TEXT, node_type TEXT)"
    )
    for r in rows:
        c.execute(
            "INSERT OR IGNORE INTO nodes VALUES (?,?,?)",
            (r[id_col], r.get(name_col, ""), r.get(type_col, "")),
        )
    c.execute("CREATE INDEX IF NOT EXISTS idx_node_name ON nodes(node_name)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_node_type ON nodes(node_type)")

    # ── edges table ────────────────────────────────────────────────────
    edges_file = os.path.join(data_dir, "TarKG_edges.csv")
    erows = _load_csv(edges_file)
    ecols = list(erows[0].keys())

    hcol = _pick(["node1", "head_id", "head", "source"], ecols, 0)
    rcol = _pick(["relation", "rel", "edge_type"], ecols, 1)
    tcol = _pick(["node2", "tail_id", "tail", "target"], ecols, 2)
    print(f"  edges mapping: head={hcol}, relation={rcol}, tail={tcol}")

    c.execute("DROP TABLE IF EXISTS edges")
    c.execute(
        "CREATE TABLE edges (head_id TEXT, relation TEXT, tail_id TEXT)"
    )
    c.executemany(
        "INSERT INTO edges VALUES (?,?,?)",
        [(r[hcol], r[rcol], r[tcol]) for r in erows],
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_head ON edges(head_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_tail ON edges(tail_id)")

    # ── node mappings ──────────────────────────────────────────────────
    map_file = os.path.join(data_dir, "TarKG_nodes_mapping.csv")
    if os.path.exists(map_file):
        mrows = _load_csv(map_file)
        mcols = list(mrows[0].keys())
        c.execute("DROP TABLE IF EXISTS node_mapping")
        c.execute(
            f"CREATE TABLE node_mapping ({', '.join(_q(col) + ' TEXT' for col in mcols)})"
        )
        c.executemany(
            f"INSERT INTO node_mapping VALUES ({','.join('?' for _ in mcols)})",
            [tuple(r[col] for col in mcols) for r in mrows],
        )
        c.execute(
            f"CREATE INDEX IF NOT EXISTS idx_map_id ON node_mapping({_q(mcols[0])})"
        )

    # ── feature tables (Drug_feature, Gene_feature, etc.) ──────────────
    for fpath in sorted(glob.glob(os.path.join(data_dir, "*_feature.csv"))):
        tname = os.path.basename(fpath).replace(".csv", "").lower()
        frows = _load_csv(fpath)
        if not frows:
            continue
        fcols = list(frows[0].keys())
        c.execute(f"DROP TABLE IF EXISTS {_q(tname)}")
        c.execute(
            f"CREATE TABLE {_q(tname)} ({', '.join(_q(col) + ' TEXT' for col in fcols)})"
        )
        c.executemany(
            f"INSERT INTO {_q(tname)} VALUES ({','.join('?' for _ in fcols)})",
            [tuple(r[col] for col in fcols) for r in frows],
        )
        c.execute(f"CREATE INDEX IF NOT EXISTS idx_{tname} ON {_q(tname)}({_q(fcols[0])})")

    conn.commit()
    conn.close()
    print(f"[TarKG] Index built → {db}")
    return db


# ── 2. Core query helpers ──────────────────────────────────────────────────
def _get_conn(data_dir: str = DATA_DIR) -> sqlite3.Connection:
    db = os.path.join(data_dir, ".tarkg_index.db")
    if not os.path.exists(db):
        build_index(data_dir)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


def _resolve_ids(conn: sqlite3.Connection, query: str) -> list[str]:
    """Resolve a query string to node IDs (exact id, exact name, or LIKE match)."""
    c = conn.cursor()
    # exact id
    c.execute("SELECT node_id FROM nodes WHERE node_id = ?", (query,))
    ids = [r["node_id"] for r in c.fetchall()]
    if ids:
        return ids
    # exact name (case-insensitive)
    c.execute(
        "SELECT node_id FROM nodes WHERE LOWER(node_name) = LOWER(?)", (query,)
    )
    ids = [r["node_id"] for r in c.fetchall()]
    if ids:
        return ids
    # partial match
    c.execute(
        "SELECT node_id FROM nodes WHERE node_name LIKE ? LIMIT 20",
        (f"%{query}%",),
    )
    return [r["node_id"] for r in c.fetchall()]


# ── 3. Public API ──────────────────────────────────────────────────────────
def query_entity(
    entity: str,
    data_dir: str = DATA_DIR,
    max_edges: int = 50,
) -> dict:
    """
    Query a single entity. Returns dict with:
      node_info, features, outgoing_edges, incoming_edges, mappings
    """
    conn = _get_conn(data_dir)
    c = conn.cursor()
    ids = _resolve_ids(conn, entity)
    if not ids:
        conn.close()
        return {"query": entity, "matched": False, "candidates": []}

    nid = ids[0]
    # node info
    c.execute("SELECT * FROM nodes WHERE node_id = ?", (nid,))
    node = dict(c.fetchone())

    # edges
    c.execute(
        "SELECT e.head_id, e.relation, e.tail_id, n.node_name AS tail_name "
        "FROM edges e LEFT JOIN nodes n ON e.tail_id = n.node_id "
        "WHERE e.head_id = ? LIMIT ?",
        (nid, max_edges),
    )
    out_edges = [dict(r) for r in c.fetchall()]

    c.execute(
        "SELECT e.head_id, e.relation, e.tail_id, n.node_name AS head_name "
        "FROM edges e LEFT JOIN nodes n ON e.head_id = n.node_id "
        "WHERE e.tail_id = ? LIMIT ?",
        (nid, max_edges),
    )
    in_edges = [dict(r) for r in c.fetchall()]

    # features (try matching feature tables by node type)
    features = {}
    ntype = node.get("node_type", "").lower()
    for tname in [f"{ntype}_feature", f"{ntype.capitalize()}_feature"]:
        try:
            c.execute(f"SELECT * FROM {_q(tname)} WHERE rowid IN "
                      f"(SELECT rowid FROM {_q(tname)} LIMIT 1)")
            sample = c.fetchone()
            if sample:
                id_col = list(dict(sample).keys())[0]
                c.execute(f"SELECT * FROM {_q(tname)} WHERE {_q(id_col)} = ?", (nid,))
                row = c.fetchone()
                if row:
                    features = dict(row)
        except sqlite3.OperationalError:
            pass

    # mappings
    mappings = []
    try:
        c.execute("SELECT * FROM node_mapping LIMIT 1")
        sample = c.fetchone()
        if sample:
            id_col = list(dict(sample).keys())[0]
            c.execute(f"SELECT * FROM node_mapping WHERE {_q(id_col)} = ?", (nid,))
            mappings = [dict(r) for r in c.fetchall()]
    except sqlite3.OperationalError:
        pass

    conn.close()
    return {
        "query": entity,
        "matched": True,
        "resolved_id": nid,
        "node_info": node,
        "features": features,
        "outgoing_edges": out_edges,
        "incoming_edges": in_edges,
        "mappings": mappings,
        "extra_ids": ids[1:5] if len(ids) > 1 else [],
    }


def query_entities(
    entities: list[str],
    data_dir: str = DATA_DIR,
    max_edges: int = 50,
) -> list[dict]:
    """Query multiple entities. Returns list of result dicts."""
    return [query_entity(e, data_dir, max_edges) for e in entities]


def get_relation_types(data_dir: str = DATA_DIR) -> list[str]:
    """List all distinct relation types."""
    conn = _get_conn(data_dir)
    c = conn.cursor()
    c.execute("SELECT DISTINCT relation FROM edges ORDER BY relation")
    rels = [r[0] for r in c.fetchall()]
    conn.close()
    return rels


def get_node_types(data_dir: str = DATA_DIR) -> list[str]:
    """List all distinct node types."""
    conn = _get_conn(data_dir)
    c = conn.cursor()
    c.execute("SELECT DISTINCT node_type FROM nodes ORDER BY node_type")
    types = [r[0] for r in c.fetchall()]
    conn.close()
    return types


def search_nodes(
    keyword: str,
    node_type: str | None = None,
    data_dir: str = DATA_DIR,
    limit: int = 20,
) -> list[dict]:
    """Search nodes by keyword, optionally filtered by type."""
    conn = _get_conn(data_dir)
    c = conn.cursor()
    if node_type:
        c.execute(
            "SELECT * FROM nodes WHERE node_name LIKE ? AND LOWER(node_type) = LOWER(?) LIMIT ?",
            (f"%{keyword}%", node_type, limit),
        )
    else:
        c.execute(
            "SELECT * FROM nodes WHERE node_name LIKE ? LIMIT ?",
            (f"%{keyword}%", limit),
        )
    results = [dict(r) for r in c.fetchall()]
    conn.close()
    return results


def get_stats(data_dir: str = DATA_DIR) -> dict:
    """Return basic statistics of the TarKG database."""
    conn = _get_conn(data_dir)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM nodes")
    n_nodes = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM edges")
    n_edges = c.fetchone()[0]
    c.execute("SELECT node_type, COUNT(*) FROM nodes GROUP BY node_type ORDER BY COUNT(*) DESC")
    type_counts = {r[0]: r[1] for r in c.fetchall()}
    conn.close()
    return {"total_nodes": n_nodes, "total_edges": n_edges, "node_type_counts": type_counts}


def query_subgraph(
    entity: str,
    n_hops: int = 1,
    data_dir: str = DATA_DIR,
    max_neighbors_per_hop: int = 50,
) -> dict:
    """
    BFS n-hop subgraph query from a seed entity.

    Returns:
      {
        "seed": "EGFR",
        "seed_id": "Gene::1956",
        "n_hops": 2,
        "nodes": { "Gene::1956": {node_id, node_name, node_type, hop}, ... },
        "edges": [ {head_id, head_name, relation, tail_id, tail_name, hop}, ... ]
      }

    hop=0 is the seed itself; hop=k means discovered at the k-th expansion.
    """
    conn = _get_conn(data_dir)
    c = conn.cursor()

    # resolve seed
    ids = _resolve_ids(conn, entity)
    if not ids:
        conn.close()
        return {"seed": entity, "matched": False}

    seed_id = ids[0]

    # BFS
    visited_nodes: dict[str, int] = {}   # node_id → hop
    all_edges: list[dict] = []
    frontier = {seed_id}
    visited_nodes[seed_id] = 0

    for hop in range(1, n_hops + 1):
        if not frontier:
            break
        placeholders = ",".join("?" for _ in frontier)
        frontier_list = list(frontier)
        next_frontier = set()

        # outgoing
        c.execute(
            f"SELECT e.head_id, e.relation, e.tail_id, "
            f"n1.node_name AS head_name, n2.node_name AS tail_name "
            f"FROM edges e "
            f"LEFT JOIN nodes n1 ON e.head_id = n1.node_id "
            f"LEFT JOIN nodes n2 ON e.tail_id = n2.node_id "
            f"WHERE e.head_id IN ({placeholders}) "
            f"LIMIT ?",
            frontier_list + [max_neighbors_per_hop * len(frontier_list)],
        )
        for r in c.fetchall():
            row = dict(r)
            row["hop"] = hop
            all_edges.append(row)
            if row["tail_id"] not in visited_nodes:
                visited_nodes[row["tail_id"]] = hop
                next_frontier.add(row["tail_id"])

        # incoming
        c.execute(
            f"SELECT e.head_id, e.relation, e.tail_id, "
            f"n1.node_name AS head_name, n2.node_name AS tail_name "
            f"FROM edges e "
            f"LEFT JOIN nodes n1 ON e.head_id = n1.node_id "
            f"LEFT JOIN nodes n2 ON e.tail_id = n2.node_id "
            f"WHERE e.tail_id IN ({placeholders}) "
            f"LIMIT ?",
            frontier_list + [max_neighbors_per_hop * len(frontier_list)],
        )
        for r in c.fetchall():
            row = dict(r)
            row["hop"] = hop
            all_edges.append(row)
            if row["head_id"] not in visited_nodes:
                visited_nodes[row["head_id"]] = hop
                next_frontier.add(row["head_id"])

        frontier = next_frontier

    # deduplicate edges
    edge_set = set()
    unique_edges = []
    for e in all_edges:
        key = (e["head_id"], e["relation"], e["tail_id"])
        if key not in edge_set:
            edge_set.add(key)
            unique_edges.append(e)

    # collect node info
    nodes_info = {}
    for nid, hop in visited_nodes.items():
        c.execute("SELECT * FROM nodes WHERE node_id = ?", (nid,))
        row = c.fetchone()
        if row:
            info = dict(row)
            info["hop"] = hop
            nodes_info[nid] = info
        else:
            nodes_info[nid] = {"node_id": nid, "hop": hop}

    conn.close()
    return {
        "seed": entity,
        "seed_id": seed_id,
        "matched": True,
        "n_hops": n_hops,
        "total_nodes": len(nodes_info),
        "total_edges": len(unique_edges),
        "nodes": nodes_info,
        "edges": unique_edges,
    }


# ── 4. Usage examples (run directly: python 41_TarKG.py) ──────────────────
def _print_subgraph(sg, show_edges=10):
    """Print a subgraph result concisely."""
    if not sg.get("matched"):
        print(f"  {sg['seed']} → ✗ not found")
        return
    print(f"  seed: {sg['seed']} → {sg['seed_id']}")
    print(f"  {sg['n_hops']}-hop subgraph: {sg['total_nodes']} nodes, {sg['total_edges']} edges")

    # group nodes by hop
    by_hop = {}
    for nid, info in sg["nodes"].items():
        h = info.get("hop", 0)
        by_hop.setdefault(h, []).append(info)
    for h in sorted(by_hop):
        names = [f"{n.get('node_name','?')}[{n.get('node_type','?')}]"
                 for n in by_hop[h][:8]]
        suffix = f" ... +{len(by_hop[h])-8}" if len(by_hop[h]) > 8 else ""
        print(f"    hop {h}: {', '.join(names)}{suffix}")

    # show sample edges
    print(f"  edges (first {min(show_edges, len(sg['edges']))}):")
    for e in sg["edges"][:show_edges]:
        h_name = e.get("head_name") or e["head_id"]
        t_name = e.get("tail_name") or e["tail_id"]
        print(f"    {h_name} —[{e['relation']}]→ {t_name}  (hop {e['hop']})")
    if len(sg["edges"]) > show_edges:
        print(f"    ... and {len(sg['edges']) - show_edges} more")


if __name__ == "__main__":
    import sys, json as _json
    if len(sys.argv) > 1:

        _cli_entities = sys.argv[1:]
        build_index()
        for _e in _cli_entities:
            _result = query_entity(_e)
            print(_json.dumps(_result, indent=2, ensure_ascii=False, default=str))
        sys.exit(0)

    # --- original demo below ---
    # 0. Build index (auto-skip if .tarkg_index.db already exists)
    print("=" * 60)
    db_path = os.path.join(DATA_DIR, ".tarkg_index.db")
    if os.path.exists(db_path):
        print(f"Step 0: Index exists → {db_path}  (skipped)")
    else:
        print("Step 0: Building SQLite index (first time only) ...")
        build_index()

    # 1. Dataset overview
    print("\n" + "=" * 60)
    print("Step 1: Dataset statistics")
    stats = get_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # 2. Peek: show a few sample nodes per type so we know what names look like
    print("\n" + "=" * 60)
    print("Step 2: Sample nodes per type")
    conn = _get_conn()
    c = conn.cursor()
    for ntype in list(stats["node_type_counts"].keys())[:6]:
        c.execute("SELECT node_id, node_name FROM nodes WHERE node_type = ? LIMIT 3", (ntype,))
        samples = [(r[0], r[1]) for r in c.fetchall()]
        print(f"  [{ntype}]  {samples}")
    conn.close()

    # 3. Pick a seed node from the most common type and do 1-hop / 2-hop
    print("\n" + "=" * 60)
    conn = _get_conn()
    c = conn.cursor()
    top_type = list(stats["node_type_counts"].keys())[0]  # most common
    c.execute("SELECT node_id, node_name FROM nodes WHERE node_type = ? LIMIT 1", (top_type,))
    row = c.fetchone()
    seed_id = row[0] if row else None
    seed_name = row[1] if row else ""
    conn.close()

    if seed_id:
        print(f"Step 3: 1-hop subgraph – {seed_name} ({top_type})")
        _print_subgraph(query_subgraph(seed_id, n_hops=1))

        print("\n" + "=" * 60)
        print(f"Step 4: 2-hop subgraph – {seed_name} ({top_type})")
        _print_subgraph(query_subgraph(seed_id, n_hops=2))

    # 5. Search example
    print("\n" + "=" * 60)
    print("Step 5: Search 'diabetes' (top 5)")
    for n in search_nodes("diabetes", limit=5):
        print(f"  {n['node_id']}\t{n['node_name']}\t{n['node_type']}")