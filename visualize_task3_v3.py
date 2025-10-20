

import argparse
import os
from datetime import datetime
import textwrap

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



SQL_A = """\
SELECT place AS mesto, COUNT(*) AS broj
FROM {dst}
GROUP BY place
ORDER BY broj DESC, mesto ASC;
"""

SQL_B = SQL_A

SQL_C = """\
SELECT
  stars,
  COUNT(*) AS broj,
  100.0 * COUNT(*) / (SELECT COUNT(*) FROM {dst}) AS procenat
FROM {dst}
GROUP BY stars
ORDER BY stars;
"""

SQL_D = """\
SELECT bucket, COUNT(*) AS broj,
       100.0 * COUNT(*) / (SELECT COUNT(*) FROM {dst}) AS procenat
FROM (
  SELECT CASE
           WHEN price_eur_norm IS NULL THEN 'Unknown'
           WHEN price_eur_norm <= 500 THEN '<= 500'
           WHEN price_eur_norm BETWEEN 501 AND 1500 THEN '501-1500'
           WHEN price_eur_norm BETWEEN 1501 AND 3000 THEN '1501-3000'
           WHEN price_eur_norm >= 3000 THEN '>= 3000'
           ELSE 'Other'
         END AS bucket
  FROM {dst}
) t
GROUP BY bucket
ORDER BY
  CASE bucket
    WHEN '<= 500' THEN 1
    WHEN '501-1500' THEN 2
    WHEN '1501-3000' THEN 3
    WHEN '>= 3000' THEN 4
    WHEN 'Unknown' THEN 5
    ELSE 6
  END;
"""

SQL_E = """\
SELECT service_type, COUNT(*) AS broj,
       100.0 * COUNT(*) / (SELECT COUNT(*) FROM {dst}) AS procenat
FROM (
  SELECT
    CASE
      WHEN board IS NULL OR TRIM(board) = '' THEN 'Other/Unknown'
      WHEN LOWER(board) REGEXP 'ultra[[:space:]]*all[[:space:]]*inclusive' THEN 'Ultra All Inclusive'
      WHEN LOWER(board) REGEXP 'all[[:space:]]*inclusive' THEN 'All Inclusive'
      WHEN LOWER(board) REGEXP 'full[[:space:]]*board|\\bfb\\b' THEN 'Full Board'
      WHEN LOWER(board) REGEXP 'half[[:space:]]*board|\\bhb\\b|polu' THEN 'Half Board'
      WHEN LOWER(board) REGEXP 'bed[[:space:]]*&[[:space:]]*breakfast|breakfast|\\bbb\\b' THEN 'Bed & Breakfast'
      WHEN LOWER(board) REGEXP 'room[[:space:]]*only|\\bro\\b|self[[:space:]]*cater' THEN 'Room Only / Self-catering'
      ELSE 'Other/Unknown'
    END AS service_type
  FROM {dst}
) x
GROUP BY service_type
ORDER BY broj DESC, service_type ASC;
"""



def parse_args():
    p = argparse.ArgumentParser(description="Task 3 — Visualization (v3.0)")
    p.add_argument("--db-url", required=True, help="mysql+pymysql://user:@localhost:3306/travel_etl")
    p.add_argument("--dst-table", default="arrangements_clean", help="Clean tabela iz Task 2")
    p.add_argument("--out-dir", default="charts", help="Folder za PNG grafikone")
    p.add_argument("--excel-out", default="task3_visualizations.xlsx", help="Excel sa agregatima")
    p.add_argument("--sql-out", default="task3_queries.sql", help="SQL upiti (a-e)")
    p.add_argument("--md-out", default="task3_report.md", help="Markdown report")
    p.add_argument("--places-topn", type=int, default=50, help="Za (b) koliko mesta prikazati (čitljivost)")
    return p.parse_args()


def eng(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True, future=True)


def run_df(en: Engine, q: str) -> pd.DataFrame:
    return pd.read_sql(text(q), con=en)


def save_bar(data, x_col, y_col, title, xlabel, ylabel, out_path, rotate_xticks=45, figsize=(12, 7)):
    plt.figure(figsize=figsize)
    plt.bar(data[x_col].astype(str), data[y_col])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotate_xticks, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_pie(labels, sizes, title, out_path, figsize=(8, 8)):
    plt.figure(figsize=figsize)
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_sql_file(path: str, dst: str):
    blob = textwrap.dedent(f"""
    -- Task 3 queries for `{dst}` (MariaDB/XAMPP friendly)

    -- (a) 10 najzastupljenijih mesta (po broju aranžmana)
    {SQL_A.format(dst=dst)}
    -- klijent prikazuje TOP 10 iz rezultata

    -- (b) broj aranžmana po mestima (full tabela, često velika)
    {SQL_B.format(dst=dst)}

    -- (c) broj i procenat po zvezdicama
    {SQL_C.format(dst=dst)}

    -- (d) broj i procenat po cenovnim opsezima (EUR total, price_eur_norm)
    {SQL_D.format(dst=dst)}

    -- (e) broj i procenat po tipu usluge (normalizovan board -> service_type)
    {SQL_E.format(dst=dst)}
    """).strip() + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(blob)


def write_md(md_path: str, artifacts: dict):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = f"""\
# Visualization Report (v3.0)

Generated: **{ts}**

**Source**: `arrangements_clean` (Task 2 output)  
**Note**: Prices are per **room for 2 persons**; bucket (d) uses total EUR (`price_eur_norm`).

## (a) Top 10 places by number of arrangements
![Top 10 Places]({artifacts['a']})

## (b) Arrangements by place (Top {artifacts['topn']})
![By Place TopN]({artifacts['b']})

## (c) Hotel star ratings — count and share
![Stars Bar]({artifacts['c_bar']})
![Stars Pie]({artifacts['c_pie']})

## (d) Price ranges (EUR total per room)
![Price Buckets Bar]({artifacts['d_bar']})
![Price Buckets Pie]({artifacts['d_pie']})

## (e) Service types (board) — count and share
![Service Bar]({artifacts['e_bar']})
![Service Pie]({artifacts['e_pie']})

---
Artifacts:
- Excel aggregates: `{artifacts['excel']}`
- SQL queries: `{artifacts['sql']}`
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    en = eng(args.db_url)
    dst = args.dst_table

    # (a)/(b) places count
    df_places = run_df(en, SQL_A.format(dst=dst))
    # (a) Top 10 places
    df_top10 = df_places.head(10).copy()
    chart_a = os.path.join(args.out_dir, "chart_a_top10_places.png")
    save_bar(df_top10, "mesto", "broj", "Top 10 mesta po broju aranžmana", "Mesto", "Broj aranžmana", chart_a)

    # (b) By place — top N (čitljivost)
    topn = args.places_topn
    df_topn = df_places.head(topn).copy()
    chart_b = os.path.join(args.out_dir, f"chart_b_places_all_top{topn}.png")
    save_bar(df_topn, "mesto", "broj", f"Broj aranžmana po mestima (Top {topn})", "Mesto", "Broj aranžmana", chart_b, rotate_xticks=60, figsize=(14, 8))

    # (c) Stars distribution
    df_stars = run_df(en, SQL_C.format(dst=dst))
    chart_c_bar = os.path.join(args.out_dir, "chart_c_stars_bar.png")
    save_bar(df_stars.fillna({"stars": "Unknown"}), "stars", "broj", "Broj aranžmana po zvezdicama hotela", "Zvezdice", "Broj aranžmana", chart_c_bar, rotate_xticks=0)
    chart_c_pie = os.path.join(args.out_dir, "chart_c_stars_pie.png")
    save_pie(labels=df_stars["stars"].fillna("Unknown").astype(str).tolist(),
             sizes=df_stars["procenat"].tolist(),
             title="Udeo aranžmana po zvezdicama hotela",
             out_path=chart_c_pie)

    # (d) Price buckets (EUR total per room)
    df_price = run_df(en, SQL_D.format(dst=dst))
    chart_d_bar = os.path.join(args.out_dir, "chart_d_price_buckets_bar.png")
    save_bar(df_price, "bucket", "broj", "Broj aranžmana po cenovnim opsezima (EUR ukupno - soba)", "Opseg cene (EUR)", "Broj aranžmana", chart_d_bar, rotate_xticks=0)
    chart_d_pie = os.path.join(args.out_dir, "chart_d_price_buckets_pie.png")
    save_pie(labels=df_price["bucket"].tolist(),
             sizes=df_price["procenat"].tolist(),
             title="Udeo aranžmana po cenovnim opsezima (EUR ukupno - soba)",
             out_path=chart_d_pie)

    # (e) Service types (board normalized)
    df_service = run_df(en, SQL_E.format(dst=dst))
    chart_e_bar = os.path.join(args.out_dir, "chart_e_service_bar.png")
    save_bar(df_service, "service_type", "broj", "Broj aranžmana po tipu usluge (board)", "Tip usluge", "Broj aranžmana", chart_e_bar, rotate_xticks=45, figsize=(12,7))
    chart_e_pie = os.path.join(args.out_dir, "chart_e_service_pie.png")
    save_pie(labels=df_service["service_type"].tolist(),
             sizes=df_service["procenat"].tolist(),
             title="Udeo aranžmana po tipu usluge (board)",
             out_path=chart_e_pie)

    # Excel agregati
    excel_path = args.excel_out
    with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
        df_places.to_excel(w, sheet_name="places_all", index=False)
        df_top10.to_excel(w, sheet_name="places_top10", index=False)
        df_topn.to_excel(w, sheet_name=f"places_top{topn}", index=False)
        df_stars.to_excel(w, sheet_name="stars", index=False)
        df_price.to_excel(w, sheet_name="price_buckets", index=False)
        df_service.to_excel(w, sheet_name="service_types", index=False)

    # SQL upiti
    write_sql_file(args.sql_out, dst)

    # MD report
    artifacts = {
        "a": chart_a,
        "b": chart_b,
        "c_bar": chart_c_bar,
        "c_pie": chart_c_pie,
        "d_bar": chart_d_bar,
        "d_pie": chart_d_pie,
        "e_bar": chart_e_bar,
        "e_pie": chart_e_pie,
        "excel": excel_path,
        "sql": args.sql_out,
        "topn": topn,
    }
    write_md(args.md_out, artifacts)

    print("DONE ✔")
    print(f"- Charts dir: {args.out_dir}")
    print(f"- Excel: {excel_path}")
    print(f"- SQL:   {args.sql_out}")
    print(f"- Report:{args.md_out}")


if __name__ == "__main__":
    main()
