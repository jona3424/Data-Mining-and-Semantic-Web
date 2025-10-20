

"""
Run :
  python clean_and_analyze.py --db-url "mysql+pymysql://root:@localhost:3306/travel_etl" --src-table arrangements --dst-table arrangements_clean --eur-rsd-rate 117.2 --excel-out analysis_task2.xlsx --sql-out analysis_task2_queries.sql --fix-schema --require-min 7000
"""

import argparse
from datetime import datetime
import sys, textwrap
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


SQL_A = "SELECT place AS mesto, COUNT(*) AS broj_aranzmana FROM {dst} GROUP BY place ORDER BY broj_aranzmana DESC, mesto ASC;"
SQL_B = "SELECT COUNT(*) AS br_4i5 FROM {dst} WHERE stars IN (4,5);"
SQL_C = "SELECT COUNT(*) AS br_all_inclusive FROM {dst} WHERE (is_all_inclusive = 1) OR (board REGEXP 'all[[:space:]]*inclusive');"
SQL_D = "SELECT COUNT(*) AS br_ge_12_nocenja FROM {dst} WHERE nights >= 12;"
SQL_E = "SELECT * FROM {dst} ORDER BY price_eur_norm DESC LIMIT 30;"
SQL_F = ("SELECT * FROM {dst} "
         "WHERE (is_all_inclusive = 1) OR (board REGEXP 'all[[:space:]]*inclusive') "
         "ORDER BY price_eur_norm DESC LIMIT 30;")

INSERT_CLEAN = """\
INSERT INTO {dst} (
  src_id, site, url, title, location_text, country, city, place, stars,
  departure_date, nights, room_type, board, is_all_inclusive,
  price_eur_norm, price_rsd_norm, price_eur_per_person, price_rsd_per_person,
  transport_mode, is_air, created_at
)
SELECT
  t.id AS src_id,
  t.site,
  t.url,
  t.title,
  t.location_text,
  t.country,
  t.city,
  NULLIF(TRIM(COALESCE(t.city, t.location_text)), '') AS place,
  t.stars,
  t.departure_date,
  t.nights,
  t.room_type,
  t.board,
  CASE
    WHEN t.is_all_inclusive IS NOT NULL THEN t.is_all_inclusive
    WHEN t.board REGEXP 'all[[:space:]]*inclusive' THEN 1
    ELSE 0
  END AS is_all_inclusive,
  CASE
    WHEN t.price_eur     IS NOT NULL THEN t.price_eur
    WHEN t.old_price_eur IS NOT NULL THEN t.old_price_eur
    WHEN t.price_rsd     IS NOT NULL THEN t.price_rsd / :rate
    WHEN t.old_price_rsd IS NOT NULL THEN t.old_price_rsd / :rate
    ELSE NULL
  END AS price_eur_norm,
  CASE
    WHEN t.price_eur     IS NOT NULL THEN t.price_eur * :rate
    WHEN t.old_price_eur IS NOT NULL THEN t.old_price_eur * :rate
    WHEN t.price_rsd     IS NOT NULL THEN t.price_rsd
    WHEN t.old_price_rsd IS NOT NULL THEN t.old_price_rsd
    ELSE NULL
  END AS price_rsd_norm,
  CASE
    WHEN t.price_eur     IS NOT NULL THEN t.price_eur / 2
    WHEN t.old_price_eur IS NOT NULL THEN t.old_price_eur / 2
    WHEN t.price_rsd     IS NOT NULL THEN (t.price_rsd / :rate) / 2
    WHEN t.old_price_rsd IS NOT NULL THEN (t.old_price_rsd / :rate) / 2
    ELSE NULL
  END AS price_eur_per_person,
  CASE
    WHEN t.price_eur     IS NOT NULL THEN (t.price_eur * :rate) / 2
    WHEN t.old_price_eur IS NOT NULL THEN (t.old_price_eur * :rate) / 2
    WHEN t.price_rsd     IS NOT NULL THEN t.price_rsd / 2
    WHEN t.old_price_rsd IS NOT NULL THEN t.old_price_rsd / 2
    ELSE NULL
  END AS price_rsd_per_person,
  'air' AS transport_mode,
  1     AS is_air,
  t.created_at
FROM (
  SELECT a.*,
         ROW_NUMBER() OVER (
           PARTITION BY
             a.site, a.url, a.title, a.location_text, a.country, a.city, a.stars,
             a.departure_date, a.nights, a.room_type, a.board
           ORDER BY a.created_at DESC, a.id DESC
         ) AS rn
  FROM {src} a
) AS t
WHERE
  t.rn = 1
  AND t.title IS NOT NULL AND NULLIF(TRIM(t.title), '') IS NOT NULL
  AND t.departure_date IS NOT NULL
  AND t.nights IS NOT NULL
  AND NULLIF(TRIM(COALESCE(t.city, t.location_text)), '') IS NOT NULL
  AND (
    t.price_eur IS NOT NULL OR t.old_price_eur IS NOT NULL
    OR t.price_rsd IS NOT NULL OR t.old_price_rsd IS NOT NULL
  );
"""

ALTERS_FOR_CLEAN = [
    "ALTER TABLE {dst} ADD COLUMN IF NOT EXISTS src_id BIGINT UNSIGNED NULL",
    "ALTER TABLE {dst} ADD COLUMN IF NOT EXISTS place VARCHAR(512) NULL",
    "ALTER TABLE {dst} ADD COLUMN IF NOT EXISTS price_eur_norm DECIMAL(12,2) NULL",
    "ALTER TABLE {dst} ADD COLUMN IF NOT EXISTS price_rsd_norm DECIMAL(14,2) NULL",
    "ALTER TABLE {dst} ADD COLUMN IF NOT EXISTS price_eur_per_person DECIMAL(12,2) NULL",
    "ALTER TABLE {dst} ADD COLUMN IF NOT EXISTS price_rsd_per_person DECIMAL(14,2) NULL",
    "ALTER TABLE {dst} ADD COLUMN IF NOT EXISTS transport_mode VARCHAR(16) NOT NULL DEFAULT 'air'",
    "ALTER TABLE {dst} ADD COLUMN IF NOT EXISTS is_air TINYINT(1) NOT NULL DEFAULT 1",
]

def parse_args():
    p = argparse.ArgumentParser(description="Clean + Analyze (v2.3) — XAMPP/MariaDB-friendly")
    p.add_argument("--db-url", required=True)
    p.add_argument("--src-table", default="arrangements")
    p.add_argument("--dst-table", default="arrangements_clean")
    p.add_argument("--eur-rsd-rate", type=float, default=117.2)
    p.add_argument("--excel-out", default="analysis_task2.xlsx")
    p.add_argument("--sql-out", default="analysis_task2_queries.sql")
    p.add_argument("--require-min", type=int, default=7000)
    p.add_argument("--fix-schema", action="store_true")
    return p.parse_args()

def eng(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True, future=True)

def ensure_schema(en: Engine, dst: str, fix: bool):
    if not fix:
        return
    with en.begin() as c:
        for stmt in ALTERS_FOR_CLEAN:
            try:
                c.execute(text(stmt.format(dst=dst)))
            except Exception:
                pass
        for idx in [
            f"CREATE INDEX idx_{dst}_place ON {dst}(place)",
            f"CREATE INDEX idx_{dst}_stars ON {dst}(stars)",
            f"CREATE INDEX idx_{dst}_nights ON {dst}(nights)",
            f"CREATE INDEX idx_{dst}_departure ON {dst}(departure_date)",
            f"CREATE INDEX idx_{dst}_board ON {dst}(board)",
            f"CREATE INDEX idx_{dst}_price_eur_norm ON {dst}(price_eur_norm)"
        ]:
            try:
                c.execute(text(idx))
            except Exception:
                pass

def populate_clean(en: Engine, src: str, dst: str, rate: float):
    with en.begin() as c:
        # 1) TRUNCATE prije inserta u bazu
        c.execute(text(f"TRUNCATE {dst}"))
        c.execute(text(INSERT_CLEAN.format(src=src, dst=dst)), {"rate": rate})

def run_df(en: Engine, q: str) -> pd.DataFrame:
    return pd.read_sql(text(q), con=en)

def export_excel(dfs: dict, path: str):
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, df in dfs.items():
            sheet = (name[:31]
                     .replace(":", "_").replace("/", "_")
                     .replace("\\", "_").replace("*", "_").replace("?", "_"))
            df.to_excel(w, sheet_name=sheet, index=False)

def write_sql_file(path: str, dst: str):
    blob = textwrap.dedent(f"""
    -- Auto-generated queries for `{dst}` (MariaDB/XAMPP friendly)

    -- (a)
    {SQL_A.format(dst=dst)}

    -- (b)
    {SQL_B.format(dst=dst)}

    -- (c)
    {SQL_C.format(dst=dst)}

    -- (d)
    {SQL_D.format(dst=dst)}

    -- (e)
    {SQL_E.format(dst=dst)}

    -- (f)
    {SQL_F.format(dst=dst)}
    """).strip() + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(blob)


def main():
    args = parse_args()
    en = eng(args.db-url if hasattr(args, "db-url") else args.db_url)
    if not hasattr(args, "db_url"):
        args.db_url = getattr(args, "db-url")

    en = eng(args.db_url)

    print("[1/6] (Optional) Adjust schema …")
    ensure_schema(en, args.dst_table, args.fix_schema)

    print("[2/6] Populate clean (TRUNCATE + INSERT … SELECT) …")
    populate_clean(en, args.src_table, args.dst_table, args.eur_rsd_rate)

    print("[3/6] Count rows …")
    n = pd.read_sql(text(f"SELECT COUNT(*) n FROM {args.dst_table}"), con=en)["n"].iloc[0]
    print(f"      Clean rows: {n:,}")

    print("[4/6] Run analytics (a-f) …")
    dfs = {}
    dfs["a_broj_po_mestima"]       = run_df(en, SQL_A.format(dst=args.dst_table))
    dfs["b_broj_4_5_zvezdica"]     = run_df(en, SQL_B.format(dst=args.dst_table))
    try:
        dfs["c_broj_all_inclusive"] = run_df(en, SQL_C.format(dst=args.dst_table))
    except Exception:
        tmp = pd.read_sql(text(f"SELECT is_all_inclusive, board FROM {args.dst_table}"), con=en)
        mask = (tmp["is_all_inclusive"] == 1) | tmp["board"].fillna("").str.contains(r"all\s*inclusive", case=False, regex=True)
        dfs["c_broj_all_inclusive"] = pd.DataFrame({"br_all_inclusive": [int(mask.sum())]})
    dfs["d_broj_ge12_nocenja"]     = run_df(en, SQL_D.format(dst=args.dst_table))
    dfs["e_top30_najskuplji"]      = run_df(en, SQL_E.format(dst=args.dst_table))
    dfs["f_top30_ai_najskuplji"]   = run_df(en, SQL_F.format(dst=args.dst_table))

    print("[5/6] Export Excel …")
    export_excel(dfs, args.excel_out)

    print("[6/6] Write SQL + Markdown …")
    write_sql_file(args.sql_out, args.dst_table)
    meta = {
        "src": args.src_table,
        "dst": args.dst_table,
        "eur_rsd": args.eur_rsd_rate,
        "clean_count": int(n),
        "min_required": int(args.require_min),
        "excel_out": args.excel_out,
        "sql_out": args.sql_out,
    }

    print(f"- Clean table: {args.dst_table} ({n:,} rows)")
    print(f"- Excel: {args.excel_out}")
    print(f"- SQL:   {args.sql_out}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("FATAL:", e)
        sys.exit(1)
