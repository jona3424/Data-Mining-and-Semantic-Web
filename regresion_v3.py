
import os, math, random, re, csv, io, json
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
from collections import Counter, defaultdict

from flask import Flask, request, render_template_string, Response, jsonify

import mysql.connector
from mysql.connector import Error
import numpy as np
from urllib.parse import urlparse


DEFAULT_DB_NAME = "travel_etl"

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_USER = os.environ.get("DB_USER", "root")
DB_PASS = os.environ.get("DB_PASS", "")
DB_NAME = os.environ.get("DB_NAME", DEFAULT_DB_NAME)
DATABASE_URL = os.environ.get("DATABASE_URL")  # npr mysql+pymysql://root:@localhost:3306/travel_etl

TABLE_NAME_OVERRIDE   = os.environ.get("TABLE_NAME")        
TARGET_FIELD_OVERRIDE = os.environ.get("TARGET_FIELD")     

PRICE_CANDIDATES = [
    "price_eur_norm", "price_eur_per_person",
    "price_rsd_norm", "price_rsd_per_person",
    "price_eur", "price_rsd"
]

MIN_NIGHTS = int(os.environ.get("MIN_NIGHTS", "1"))
MAX_NIGHTS = int(os.environ.get("MAX_NIGHTS", "30"))
MIN_STARS  = int(os.environ.get("MIN_STARS", "0"))
MAX_STARS  = int(os.environ.get("MAX_STARS", "5"))

# Trening
LR         = float(os.environ.get("LR", "0.05"))
EPOCHS     = int(os.environ.get("EPOCHS", "1000"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
L2_LAMBDA  = float(os.environ.get("L2_LAMBDA", "1e-3"))
PATIENCE   = int(os.environ.get("PATIENCE", "60"))
LR_DECAY   = float(os.environ.get("LR_DECAY", "0.99"))
TEST_RATIO = float(os.environ.get("TEST_RATIO", "0.2"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))

# Preciznost
USE_LOG_TARGET      = os.environ.get("USE_LOG_TARGET", "1") == "1"
USE_TEMPORAL_SPLIT  = os.environ.get("TEMPORAL_SPLIT", "0") == "1"

CITY_MIN_FREQ       = int(os.environ.get("CITY_MIN_FREQ", "25"))
HOTEL_MIN_FREQ      = int(os.environ.get("HOTEL_MIN_FREQ", "25"))
ROOM_MIN_FREQ       = int(os.environ.get("ROOM_MIN_FREQ", "30"))
TRANSP_MIN_FREQ     = int(os.environ.get("TRANSP_MIN_FREQ", "30"))

P_LOW, P_HIGH       = float(os.environ.get("WINSOR_P_LOW", "2")), float(os.environ.get("WINSOR_P_HIGH", "98"))
LEAD_TIME_MAX       = int(os.environ.get("LEAD_TIME_MAX", "365"))

# Istorija
HIST_DEFAULT_PER = int(os.environ.get("HIST_DEFAULT_PER", "25"))
HIST_MAX_PER     = int(os.environ.get("HIST_MAX_PER", "200"))


BASE_STYLE = """
  body {font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; max-width: 1100px; margin: 2rem auto; padding: 0 1rem;}
  h1 {margin-bottom: .25rem;}
  .card {border: 1px solid #ddd; border-radius: 12px; padding: 1rem; margin-top: 1rem;}
  label {display:block; margin-top:.5rem; font-weight:600;}
  input, select {width:100%; padding:.5rem; margin-top:.25rem;}
  button {padding:.6rem 1rem; margin-top:1rem; border:0; border-radius:10px; background:#111; color:#fff; cursor:pointer;}
  .grid {display:grid; gap: 1rem; grid-template-columns: repeat(auto-fit,minmax(220px,1fr));}
  .muted {color:#666; font-size:.9rem;}
  .kpi {display:flex; gap:1rem; flex-wrap:wrap;}
  .kpi div{background:#fafafa;border:1px solid #eee;border-radius:10px;padding:.75rem 1rem;}
  .pred {font-size:1.35rem; font-weight:700; margin-top:.5rem;}
  nav {display:flex; gap:.75rem; align-items:center; flex-wrap:wrap; margin:.5rem 0 0 0}
  nav a {text-decoration:none; color:#111; background:#f3f3f3; border:1px solid #e6e6e6; border-radius:9px; padding:.4rem .7rem;}
  table {width:100%; border-collapse: collapse;}
  th, td {border-bottom:1px solid #eee; padding:.5rem .4rem; text-align:left; font-size:.92rem;}
  th {background:#fafafa;}
  .row {display:flex; gap:.75rem; flex-wrap:wrap;}
"""

HTML_TPL = """
<!doctype html>
<html lang="sr">
<head>
  <meta charset="utf-8">
  <title>Predikcija cene - v2.4</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>""" + BASE_STYLE + """</style>
</head>
<body>
  <h1>Predikcija cene aranžmana</h1>
  <nav>
    <a href="/">🏠 Početna</a>
    <a href="/history">🕘 Istorija</a>
    <a href="/history.csv">⬇️ Izvoz (CSV)</a>
  </nav>

  <div class="card">
    <div class="muted">
      <b>Tabla:</b> {{ meta.table }} &nbsp;|&nbsp;
      <b>Cilj:</b> {{ meta.target }} &nbsp;|&nbsp;
      <b>Valuta:</b> {{ currency }} &nbsp;|&nbsp;
      <b>Log-target:</b> {{ "DA" if meta.log_target else "NE" }} &nbsp;|&nbsp;
      <b>Split:</b> {{ "Temporalni" if meta.temporal else "Nasumični" }}
    </div>
  </div>

  <div class="card">
    <form method="post" action="/predict">
      <div class="grid">
        <div><label>Grad (city)</label>
          <select name="city" required>{% for c in cities %}<option value="{{c}}">{{c}}</option>{% endfor %}</select>
        </div>
        <div><label>Broj noćenja</label><input type="number" name="nights" value="7" min="1" max="30" required></div>
        <div><label>Broj zvezdica</label>
          <select name="stars" required>{% for s in [0,1,2,3,4,5] %}<option value="{{s}}">{{s}}</option>{% endfor %}</select>
        </div>
        <div><label>Usluga (board)</label>
          <select name="board" required>{% for b in boards %}<option value="{{b}}">{{b}}</option>{% endfor %}</select>
        </div>
        <div><label>Transport</label>
          <select name="transport" required>{% for t in transports %}<option value="{{t}}">{{t}}</option>{% endfor %}</select>
        </div>
        <div><label>Tip sobe</label>
          <select name="room" required>{% for r in rooms %}<option value="{{r}}">{{r}}</option>{% endfor %}</select>
        </div>
        <div><label>Mesec (1-12)</label>
          <select name="month" required>{% for m in range(1,13) %}<option value="{{m}}">{{m}}</option>{% endfor %}</select>
        </div>
      </div>
      <button type="submit">Predvidi cenu</button>
    </form>
  </div>

  {% if prediction is not none %}
  <div class="card">
    <div>Prediktivna vrednost:</div>
    <div class="pred">{{ "%.2f" % prediction }} {{ currency }}</div>
    <div class="muted">Linearni model (ručno), early-stop po val MAE (EUR). Rezultat je upisan u istoriju.</div>
  </div>
  {% endif %}

  <div class="card">
    <h3>Evaluacija (test skup)</h3>
    <div class="kpi">
      <div><b>MAE</b><br>{{ "%.2f" % metrics.mae }}</div>
      <div><b>RMSE</b><br>{{ "%.2f" % metrics.rmse }}</div>
      <div><b>MAPE</b><br>{{ "%.2f" % metrics.mape }}%</div>
      <div><b>R²</b><br>{{ "%.4f" % metrics.r2 }}</div>
      <div><b>Train</b><br>{{ counts.train }}</div>
      <div><b>Test</b><br>{{ counts.test }}</div>
      <div><b>#features</b><br>{{ counts.features }}</div>
      <div><b>Baseline MAE</b><br>{{ "%.2f" % metrics.baseline_mae }}</div>
    </div>
  </div>
</body>
</html>
"""

HTML_HIST = """
<!doctype html>
<html lang="sr">
<head>
  <meta charset="utf-8">
  <title>Istorija pretraga i predikcija</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>""" + BASE_STYLE + """</style>
</head>
<body>
  <h1>Istorija pretraga i predikcija</h1>
  <nav>
    <a href="/">🏠 Početna</a>
    <a href="/history.csv">⬇️ Izvoz (CSV)</a>
    <a href="/api/history">🧩 API (JSON)</a>
  </nav>

  <div class="card">
    <form method="get" class="row">
      <div style="min-width:190px;">
        <label>Grad</label>
        <select name="city">
          <option value="">(svi)</option>
          {% for c in cities %}<option value="{{c}}" {% if c==flt.city %}selected{% endif %}>{{c}}</option>{% endfor %}
        </select>
      </div>
      <div style="min-width:140px;">
        <label>Mesec</label>
        <select name="month">
          <option value="">(svi)</option>
          {% for m in range(1,13) %}<option value="{{m}}" {% if flt.month == m %}selected{% endif %}>{{m}}</option>{% endfor %}
        </select>
      </div>
      <div style="min-width:140px;">
        <label>Board</label>
        <select name="board">
          <option value="">(svi)</option>
          {% for b in boards %}<option value="{{b}}" {% if b==flt.board %}selected{% endif %}>{{b}}</option>{% endfor %}
        </select>
      </div>
      <div style="min-width:160px;">
        <label>Po stranici</label>
        <input type="number" name="per" value="{{per}}" min="1" max="200">
      </div>
      <div style="min-width:160px;">
        <label>&nbsp;</label>
        <button type="submit">Primeni filtere</button>
      </div>
    </form>
  </div>

  <div class="card">
    <div class="muted">Strana {{page}} / {{pages}} &nbsp;•&nbsp; Ukupno: {{total}} zapisa</div>
    <div style="overflow:auto;">
      <table>
        <thead>
          <tr>
            <th>#</th><th>Vreme</th><th>Grad</th><th>Hotel</th><th>Noćenja</th><th>Zvezd.</th>
            <th>Board</th><th>Transport</th><th>Soba</th><th>Mesec</th>
            <th>Predikcija</th><th>Valuta</th><th>IP</th><th>UA (skraćeno)</th>
          </tr>
        </thead>
        <tbody>
          {% for it in items %}
          <tr>
            <td>{{it.id}}</td>
            <td>{{it.created_at}}</td>
            <td>{{it.city}}</td>
            <td>{{it.hotel}}</td>
            <td>{{it.nights}}</td>
            <td>{{it.stars}}</td>
            <td>{{it.board}}</td>
            <td>{{it.transport}}</td>
            <td>{{it.room}}</td>
            <td>{{it.month}}</td>
            <td>{{"%.2f" % it.prediction}}</td>
            <td>{{it.currency}}</td>
            <td>{{it.ip}}</td>
            <td>{{it.user_agent_short}}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    <div class="row" style="margin-top: .75rem">
      {% if page>1 %}
        <a href="{{pager_url(page-1)}}" style="text-decoration:none;">⬅️ Prethodna</a>
      {% endif %}
      {% if page<pages %}
        <a href="{{pager_url(page+1)}}" style="text-decoration:none;">Sledeća ➡️</a>
      {% endif %}
    </div>
  </div>
</body>
</html>
"""



def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)

def connect_db():
    if DATABASE_URL:
        parsed = urlparse(DATABASE_URL)
        user = parsed.username or DB_USER
        pwd  = parsed.password or DB_PASS
        host = parsed.hostname or DB_HOST
        port = parsed.port or 3306
        db   = (parsed.path or "/").lstrip("/") or DB_NAME
        return mysql.connector.connect(host=host, port=port, user=user, password=pwd, database=db)
    return mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME)

def fetch_all(sql: str, params: Optional[tuple] = None) -> list:
    cnx = connect_db()
    try:
        cur = cnx.cursor(); cur.execute(sql, params or ()); rows = cur.fetchall(); cur.close(); return rows
    finally:
        cnx.close()

def fetch_dict(sql: str, params: Optional[tuple] = None) -> list:
    cnx = connect_db()
    try:
        cur = cnx.cursor(dictionary=True); cur.execute(sql, params or ()); rows = cur.fetchall(); cur.close(); return rows
    finally:
        cnx.close()

def exec_write(sql: str, params: tuple) -> int:
    """Execute INSERT/UPDATE/DELETE, return lastrowid if any."""
    cnx = connect_db()
    try:
        cur = cnx.cursor()
        cur.execute(sql, params)
        last_id = cur.lastrowid or 0
        cnx.commit()
        cur.close()
        return int(last_id)
    finally:
        cnx.close()

def table_exists(db: str, table: str) -> bool:
    return len(fetch_all("SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s", (db, table)))>0

def list_columns(db: str, table: str) -> List[str]:
    return [r[0] for r in fetch_all("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s", (db, table))]

def has_non_null(table: str, col: str) -> bool:
    try:
        rows = fetch_all(f"SELECT COUNT(1) FROM `{table}` WHERE `{col}` IS NOT NULL LIMIT 1")
        return (rows[0][0] if rows else 0) > 0
    except Error:
        return False

def current_db_name() -> str:
    cnx = connect_db()
    try: return cnx.database
    finally: cnx.close()

def pick_table_and_target() -> Tuple[str, str]:
    db = current_db_name()
    need = {"city","board","nights","stars","departure_date","title"}
    if TABLE_NAME_OVERRIDE and table_exists(db, TABLE_NAME_OVERRIDE):
        cols = set(list_columns(db, TABLE_NAME_OVERRIDE))
        if not need.issubset(cols): raise RuntimeError(f"Tabela '{TABLE_NAME_OVERRIDE}' nema obavezne kolone {need}.")
        if TARGET_FIELD_OVERRIDE and TARGET_FIELD_OVERRIDE in cols and has_non_null(TABLE_NAME_OVERRIDE, TARGET_FIELD_OVERRIDE):
            return TABLE_NAME_OVERRIDE, TARGET_FIELD_OVERRIDE
        for cand in PRICE_CANDIDATES:
            if cand in cols and has_non_null(TABLE_NAME_OVERRIDE, cand):
                return TABLE_NAME_OVERRIDE, cand
    for t in ["arrangements_clean","arrangements"]:
        if table_exists(db, t):
            cols = set(list_columns(db, t))
            if need.issubset(cols):
                for cand in ([TARGET_FIELD_OVERRIDE] if TARGET_FIELD_OVERRIDE else [])+PRICE_CANDIDATES:
                    if cand and cand in cols and has_non_null(t,cand): return t, cand
    for (tname,) in fetch_all("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA=%s",(db,)):
        cols = set(list_columns(db, tname))
        if need.issubset(cols):
            for cand in PRICE_CANDIDATES:
                if cand in cols and has_non_null(tname, cand): return tname, cand
    raise RuntimeError("Nisam našao validnu (tabela, target) kombinaciju.")


BOARD_MAP_PATTERNS = [
    (r"\broom\s*only\b|\bro\b", "ROOM_ONLY"),
    (r"\bself\s*cater|\bapartment\b|\bapt\b", "SELF_CATERING"),
    (r"\bbed\s*&?\s*breakfast\b|\bbb\b|no[cć]enje.*doru", "BED_BREAKFAST"),
    (r"\bhalf\s*board\b|\bhb\b|polupansi", "HALF_BOARD"),
    (r"\bfull\s*board\b|\bfb\b", "FULL_BOARD"),
    (r"\bultra\s*all\b|\buai\b", "ULTRA_AI"),
    (r"\b(all\s*in(c(lusive)?)?|ai\s*plus|all\s*inclusive\s*plus)\b", "ALL_INCLUSIVE"),
    (r"\b(premium\s*all|elite\s*all|all\s*in\s*concept)\b", "AI_PREMIUM"),
]

def normalize_board(board: Optional[str]) -> str:
    s = (board or "").strip().lower()
    if not s: return "BOARD_OTHER"
    for pat, lab in BOARD_MAP_PATTERNS:
        if re.search(pat, s, re.I): return lab
    return "BOARD_OTHER"

def normalize_room(room: Optional[str]) -> str:
    if not room: return "ROOM_OTHER"
    s = str(room).lower()
    if "suite" in s: return "ROOM_SUITE"
    if "studio" in s: return "ROOM_STUDIO"
    if "family" in s: return "ROOM_FAMILY"
    if "apartment" in s or "apt" in s: return "ROOM_APART"
    if "standard" in s or "std" in s: return "ROOM_STD"
    if "deluxe" in s or "dlx" in s: return "ROOM_DELUXE"
    return "ROOM_OTHER"

def safe_dt(x):
    if x is None: return None
    if isinstance(x, datetime): return x
    try: return datetime.fromisoformat(str(x))
    except: return None

def winsorize_array(values: np.ndarray, low=2, high=98):
    lo, hi = np.percentile(values, [low, high])
    return np.clip(values, lo, hi), float(lo), float(hi)

def rare_bucket(vals: List[str], min_freq: int, other_token: str) -> List[str]:
    cnt = Counter(vals)
    return [ (v if cnt[v] >= min_freq else other_token) for v in vals ]

def compute_smoothed_mean(rows: List[Dict[str,Any]], key: str, price_key: str, m: float, global_mean: float) -> Dict[str, float]:
    by = defaultdict(list)
    for r in rows: by[str(r.get(key,""))].append(r[price_key])
    means={}
    for k,vals in by.items():
        n=len(vals)
        means[k] = (float(np.sum(vals)) + m*global_mean) / (n + m)
    means["OTHER_"+key.upper()] = global_mean
    return means


def fetch_rows(table: str, target: str) -> List[Dict[str,Any]]:
    cols = "title, city, place, board, room_type, stars, nights, departure_date, created_at, transport_mode, is_air, `{}` AS price".format(target)
    sql=f"""
      SELECT {cols}
      FROM `{table}`
      WHERE `{target}` IS NOT NULL
        AND city IS NOT NULL AND city <> ''
        AND board IS NOT NULL AND board <> ''
        AND nights BETWEEN %s AND %s
        AND stars BETWEEN %s AND %s
    """
    rows = fetch_dict(sql, (MIN_NIGHTS, MAX_NIGHTS, MIN_STARS, MAX_STARS))
    out=[]
    for r in rows:
        try:
            r["price"]=float(r["price"])
            r["nights"]=int(r["nights"])
            r["stars"]=int(r["stars"])
            r["title"]=str(r.get("title") or "").strip() or "OTHER_HOTEL"
            r["city"]=str(r.get("city") or "").strip()
            r["place"]=str(r.get("place") or "").strip() or "OTHER_PLACE"
            r["board_cat"]=normalize_board(r.get("board"))
            r["room_group"]=normalize_room(r.get("room_type"))
            r["transport_mode"]=str(r.get("transport_mode") or "").strip().lower() or "other"
            r["is_air"]=int(r.get("is_air")) if r.get("is_air") is not None else 1
            dep=safe_dt(r.get("departure_date")); r["month"]=int(dep.month) if dep else 0
            r["created_at"]=safe_dt(r.get("created_at"))
            r["lead_time"]=0
            if dep and r["created_at"]:
                r["lead_time"]=max(0, min(LEAD_TIME_MAX, (dep - r["created_at"]).days))
            r["stars_unknown"]=1 if (r["stars"]==0) else 0
            out.append(r)
        except Exception:
            continue
    return out


class OneHotEncoderManual:
    def __init__(self): self.vocabs: Dict[str,List[str]] = {}
    def fit(self, rows: List[Dict[str,Any]], cols: List[str]):
        for c in cols:
            vals = sorted({str(r.get(c,"")).strip() for r in rows})
            self.vocabs[c] = list(vals)
    def transform(self, rows: List[Dict[str,Any]], cols: List[str]) -> np.ndarray:
        parts=[]
        for c in cols:
            vocab = self.vocabs.get(c, [])
            idx = {v:i for i,v in enumerate(vocab)}
            M = np.zeros((len(rows), len(vocab)), dtype=float)
            for i,r in enumerate(rows):
                key = str(r.get(c,"")).strip()
                if key in idx: M[i, idx[key]] = 1.0
            parts.append(M)
        return np.concatenate(parts, axis=1) if parts else np.zeros((len(rows),0),dtype=float)
    def names(self, cols: List[str]) -> List[str]:
        out=[]
        for c in cols:
            for v in self.vocabs.get(c, []):
                out.append(f"{c}={v}")
        return out

class StandardScalerManual:
    def __init__(self): self.means={}; self.stds={}
    def fit(self, rows: List[Dict[str,Any]], cols: List[str]):
        for c in cols:
            arr = np.array([float(r.get(c,0.0)) for r in rows], dtype=float)
            mu = float(arr.mean()) if arr.size else 0.0
            sd = float(arr.std()) if arr.size else 1.0
            if sd <= 1e-12: sd = 1.0
            self.means[c], self.stds[c] = mu, sd
    def transform(self, rows: List[Dict[str,Any]], cols: List[str]) -> np.ndarray:
        X = np.zeros((len(rows), len(cols)), dtype=float)
        for j,c in enumerate(cols):
            mu = self.means.get(c,0.0); sd = self.stds.get(c,1.0)
            if sd<=1e-12: sd=1.0
            for i,r in enumerate(rows):
                x = float(r.get(c,0.0)); X[i,j]=(x-mu)/sd
        return X

#  MODEL (GD + early-stop po val MAE EUR)
class LinearRegressionGD:
    def __init__(self, lr=0.05, epochs=800, batch_size=128, l2_lambda=1e-3, patience=40, lr_decay=1.0, verbose=False):
        self.lr=lr; self.epochs=epochs; self.batch=batch_size
        self.l2=l2_lambda; self.patience=patience; self.lr_decay=lr_decay
        self.verbose=verbose
        self.w: Optional[np.ndarray]=None
    @staticmethod
    def _add_bias(X): return np.hstack([X, np.ones((X.shape[0],1),dtype=float)])
    def fit(self, X: np.ndarray, y: np.ndarray, eval_fn=None):
        Xb=self._add_bias(X); n,d=Xb.shape; self.w=np.zeros(d,dtype=float)
        best=float("inf"); best_w=self.w.copy(); patience=self.patience; lr=self.lr
        for ep in range(self.epochs):
            idx=np.random.permutation(n); Xb=Xb[idx]; y=y[idx]
            for s in range(0,n,self.batch):
                e=min(s+self.batch,n); Xbt=Xb[s:e]; yt=y[s:e]
                yhat=Xbt@self.w
                grad=(2.0/(e-s))*(Xbt.T@(yhat-yt))
                reg=2.0*self.l2*self.w; reg[-1]=0.0; grad+=reg
                self.w-=lr*grad
            lr*=self.lr_decay
            score=float(eval_fn(self.w)) if eval_fn is not None else float(np.mean((self._add_bias(X)@self.w - y)**2))
            if score+1e-9<best: best=score; best_w=self.w.copy(); patience=self.patience
            else:
                patience-=1
                if patience<=0:
                    if self.verbose: print(f"[early stop] epoch={ep} best={best:.4f}")
                    break
        self.w=best_w
    def predict(self,X): return self._add_bias(X)@self.w

#  METRIKE
def mae(y,yh): return float(np.mean(np.abs(y-yh)))
def rmse(y,yh): return float(np.sqrt(np.mean((y-yh)**2)))
def mape(y,yh, min_y=50.0):
    denom = np.maximum(np.abs(y), min_y)
    return float(np.mean(np.abs((y-yh)/denom)) * 100.0)
def r2_score(y,yh):
    ss_res=float(np.sum((y-yh)**2)); ss_tot=float(np.sum((y-np.mean(y))**2))
    return 1.0-(ss_res/ss_tot if ss_tot>1e-12 else 0.0)


class Pipeline:
    def __init__(self, table: str, target: str):
        self.table=table; self.target=target
        # kategorije (posle bucketa)
        self.cat_cols=["city","place","transport_mode","board_cat","room_group","hotel_title"]
        # numerički
        self.num_cols=[
            "nights","stars","stars_unknown","is_air",
            "lead_time","is_peak","month_sin","month_cos",
            "nights2","stars2","nights_x_stars",
            "hotel_mean_price","city_mean_price"
        ]
        self.enc=OneHotEncoderManual(); self.scaler=StandardScalerManual()
        self.model=LinearRegressionGD(lr=LR,epochs=EPOCHS,batch_size=BATCH_SIZE,l2_lambda=L2_LAMBDA,
                                      patience=PATIENCE,lr_decay=LR_DECAY,verbose=False)
        self.currency="EUR" if "eur" in target.lower() else "RSD"
        self._winsor_lohi=None
        self._hotel_means={}; self._city_means={}; self._global_mean=0.0

    def _pre_bucket(self, rows: List[Dict[str,Any]]):
        cities  = rare_bucket([r["city"] for r in rows],  CITY_MIN_FREQ,  "OTHER_CITY")
        places  = rare_bucket([r["place"] for r in rows], CITY_MIN_FREQ,  "OTHER_PLACE")
        rooms   = rare_bucket([r["room_group"] for r in rows], ROOM_MIN_FREQ,"ROOM_OTHER")
        trans   = rare_bucket([r["transport_mode"] for r in rows], TRANSP_MIN_FREQ,"other")
        hotels  = rare_bucket([r["title"] for r in rows], HOTEL_MIN_FREQ,"OTHER_HOTEL")
        for i,r in enumerate(rows):
            r["city"]=cities[i]; r["place"]=places[i]
            r["room_group"]=rooms[i]; r["transport_mode"]=trans[i]
            r["hotel_title"]=hotels[i]

    def _add_derived(self, rows: List[Dict[str,Any]]):
        for r in rows:
            ang=2*math.pi*((int(r["month"])%12)/12.0)
            r["month_sin"]=math.sin(ang); r["month_cos"]=math.cos(ang)
            r["is_peak"]=1 if int(r["month"]) in (7,8) else 0
            r["nights2"]=r["nights"]**2
            r["stars2"]=r["stars"]**2
            r["nights_x_stars"]=r["nights"]*r["stars"]

    def _fit_means(self, rows_train: List[Dict[str,Any]]):
        prices = np.array([r["price"] for r in rows_train], dtype=float)
        prices_clip, lo, hi = winsorize_array(prices, P_LOW, P_HIGH)
        self._winsor_lohi=(lo,hi)
        self._global_mean=float(np.mean(prices_clip))
        temp=[]
        for i,r in enumerate(rows_train):
            rr=r.copy(); rr["price_clip"]=float(prices_clip[i]); temp.append(rr)
        self._hotel_means = compute_smoothed_mean(temp, "hotel_title", "price_clip", m=60.0, global_mean=self._global_mean)
        self._city_means  = compute_smoothed_mean(temp, "city",        "price_clip", m=50.0, global_mean=self._global_mean)

    def _attach_means(self, rows: List[Dict[str,Any]]):
        g=self._global_mean if self._global_mean>0 else float(np.mean([r["price"] for r in rows]))
        for r in rows:
            r["hotel_mean_price"]=float(self._hotel_means.get(r["hotel_title"], g))
            r["city_mean_price"] =float(self._city_means.get(r["city"], g))

    def _build_Xy(self, rows: List[Dict[str,Any]], log_target: bool) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        y_raw = np.array([float(r["price"]) for r in rows], dtype=float)
        if self._winsor_lohi is None:
            y_clip, lo, hi = winsorize_array(y_raw, P_LOW, P_HIGH); self._winsor_lohi=(lo,hi)
        else:
            lo,hi=self._winsor_lohi; y_clip=np.clip(y_raw, lo, hi)
        y_t = np.log(np.maximum(y_clip,1e-6)) if log_target else y_clip

        X_cat = self.enc.transform(rows, self.cat_cols)
        X_num = self.scaler.transform(rows, self.num_cols)
        X = np.hstack([X_cat, X_num]) if X_cat.size else X_num
        return X, y_t, y_clip

    def fit(self, rows: List[Dict[str,Any]], log_target: bool, temporal_split: bool):
        self._pre_bucket(rows)
        self._add_derived(rows)

        # split
        if temporal_split:
            rows_sorted = sorted(rows, key=lambda r: (int(r["month"]), r["nights"]))
            split = int(len(rows_sorted)*(1.0-TEST_RATIO))
            rows_train, rows_test = rows_sorted[:split], rows_sorted[split:]
        else:
            idx=np.arange(len(rows)); np.random.shuffle(idx); split=int(len(rows)*(1.0-TEST_RATIO))
            rows_train=[rows[i] for i in idx[:split]]; rows_test=[rows[i] for i in idx[split:]]

        # means iz treninga  
        self._fit_means(rows_train)
        self._attach_means(rows_train); self._attach_means(rows_test)

        # fit enc/scaler na treningu
        self.enc.fit(rows_train, self.cat_cols)
        self.scaler.fit(rows_train, self.num_cols)

        X_train, y_train_t, y_train_clip = self._build_Xy(rows_train, log_target)
        X_test,  y_test_t,  y_test_clip  = self._build_Xy(rows_test,  log_target)

        # valid set
        if X_train.shape[0]>10:
            v=max(1,int(0.1*X_train.shape[0]))
            X_val, y_val_t, y_val_clip = X_train[-v:], y_train_t[-v:], y_train_clip[-v:]
            X_tr, y_tr_t = X_train[:-v], y_train_t[:-v]
        else:
            X_tr, y_tr_t = X_train, y_train_t
            X_val, y_val_t, y_val_clip = None, None, None

        def eval_fn(w):
            if X_val is None:
                yhat = self.model._add_bias(X_tr) @ w
                return float(np.mean((yhat - y_tr_t)**2))
            yhat_t = self.model._add_bias(X_val) @ w
            yhat = np.exp(yhat_t) if log_target else yhat_t
            ytrue = y_val_clip if log_target else y_val_t
            return mae(ytrue, yhat)

        self.model.fit(X_tr, y_tr_t, eval_fn=eval_fn)

        # test evaluacija u EUR
        y_pred_t = self.model.predict(X_test)
        y_pred = np.exp(y_pred_t) if log_target else y_pred_t
        y_true = y_test_clip if log_target else y_test_t

        metrics = {
            "mae": mae(y_true,y_pred),
            "rmse": rmse(y_true,y_pred),
            "mape": mape(y_true,y_pred),
            "r2": r2_score(y_true,y_pred),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
        }
        metrics["baseline_mae"]=mae(y_true, np.full_like(y_true, float(np.median(y_true))))
        return metrics, rows_train, rows_test

    def predict_one(self, form: Dict[str,Any], log_target: bool) -> float:
        row={
            "title": str(form.get("hotel") or "ANY_HOTEL"),
            "city": str(form["city"]).strip(),
            "place": "OTHER_PLACE",
            "board_cat": str(form["board"]).strip(),
            "room_group": str(form["room"]).strip(),
            "transport_mode": str(form["transport"]).strip().lower(),
            "is_air": 1 if str(form["transport"]).strip().lower()=="air" else 0,
            "nights": int(form["nights"]),
            "stars": int(form["stars"]),
            "stars_unknown": 1 if int(form["stars"])==0 else 0,
            "price": 0.0,
            "month": int(form["month"]),
            "lead_time": 0
        }
        for col, other in [("city","OTHER_CITY"),("place","OTHER_PLACE"),("room_group","ROOM_OTHER"),
                           ("transport_mode","other"),("title","OTHER_HOTEL")]:
            if row[col] not in self.enc.vocabs.get(col, []): row[col]=other

        ang=2*math.pi*((row["month"]%12)/12.0)
        row["month_sin"]=math.sin(ang); row["month_cos"]=math.cos(ang)
        row["is_peak"]=1 if row["month"] in (7,8) else 0
        row["nights2"]=row["nights"]**2
        row["stars2"]=row["stars"]**2
        row["nights_x_stars"]=row["nights"]*row["stars"]

        g=self._global_mean if self._global_mean>0 else 0.0
        row["hotel_mean_price"]=float(self._hotel_means.get(row["title"], g))
        row["city_mean_price"]=float(self._city_means.get(row["city"], g))

        X_cat = self.enc.transform([row], self.cat_cols)
        X_num = self.scaler.transform([row], self.num_cols)
        X = np.hstack([X_cat,X_num]) if X_cat.size else X_num

        y_hat_t = float(self.model.predict(X)[0])
        y_hat = float(np.exp(y_hat_t)) if USE_LOG_TARGET else y_hat_t
        return max(0.0, y_hat)


HIST_TABLE = "prediction_history"

def ensure_history_table():
    db = current_db_name()
    if table_exists(db, HIST_TABLE):
        return
    sql = f"""
    CREATE TABLE IF NOT EXISTS `{HIST_TABLE}` (
      `id` BIGINT AUTO_INCREMENT PRIMARY KEY,
      `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      `ip` VARCHAR(64) NULL,
      `user_agent` VARCHAR(512) NULL,
      `city` VARCHAR(128) NOT NULL,
      `hotel` VARCHAR(256) NULL,
      `nights` INT NOT NULL,
      `stars` INT NOT NULL,
      `board` VARCHAR(64) NOT NULL,
      `transport` VARCHAR(64) NOT NULL,
      `room` VARCHAR(64) NOT NULL,
      `month` INT NOT NULL,
      `prediction` DOUBLE NOT NULL,
      `currency` VARCHAR(8) NOT NULL,
      `model_table` VARCHAR(128) NOT NULL,
      `model_target` VARCHAR(128) NOT NULL,
      `log_target` TINYINT(1) NOT NULL,
      `temporal_split` TINYINT(1) NOT NULL,
      `mae` DOUBLE NULL,
      `rmse` DOUBLE NULL,
      `mape` DOUBLE NULL,
      `r2` DOUBLE NULL,
      INDEX (`created_at`),
      INDEX (`city`),
      INDEX (`month`),
      INDEX (`board`),
      INDEX (`transport`)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    exec_write(sql, ())

def get_client_ip() -> str:
    xf = request.headers.get("X-Forwarded-For", "")
    if xf:
        return xf.split(",")[0].strip()
    return request.remote_addr or ""

def insert_history_row(form: Dict[str, Any], prediction: float, currency: str, metrics: Dict[str, Any], meta: Dict[str, Any]) -> int:
    ua = request.headers.get("User-Agent", "")[:512]
    ip = get_client_ip()[:64]
    sql = f"""
    INSERT INTO `{HIST_TABLE}`
    (ip, user_agent, city, hotel, nights, stars, board, transport, room, month,
     prediction, currency, model_table, model_target, log_target, temporal_split, mae, rmse, mape, r2)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    params = (
        ip, ua, str(form.get("city") or "")[:128], str(form.get("hotel") or "")[:256],
        int(form.get("nights") or 0), int(form.get("stars") or 0),
        str(form.get("board") or "")[:64], str(form.get("transport") or "")[:64],
        str(form.get("room") or "")[:64], int(form.get("month") or 0),
        float(prediction), currency,
        str(meta.get("table") or "")[:128], str(meta.get("target") or "")[:128],
        1 if meta.get("log_target") else 0, 1 if meta.get("temporal") else 0,
        float(metrics.get("mae") or 0), float(metrics.get("rmse") or 0),
        float(metrics.get("mape") or 0), float(metrics.get("r2") or 0)
    )
    return exec_write(sql, params)

def build_history_filters(args) -> Tuple[str, list]:
    where = []
    params: list = []
    city = args.get("city")
    month = args.get("month")
    board = args.get("board")
    if city:
        where.append("city = %s"); params.append(city)
    if month:
        try:
            m = int(month)
            if 1 <= m <= 12:
                where.append("month = %s"); params.append(m)
        except:
            pass
    if board:
        where.append("board = %s"); params.append(board)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    return clause, params

def query_history(args, page: int, per: int):
    clause, params = build_history_filters(args)
    # total
    total = fetch_all(f"SELECT COUNT(*) FROM `{HIST_TABLE}` {clause}", tuple(params))[0][0]
    pages = max(1, (total + per - 1)//per)
    page = min(max(1, page), pages)
    off = (page-1)*per
    rows = fetch_dict(f"""
      SELECT id, created_at, ip, user_agent, city, hotel, nights, stars, board, transport, room, month, prediction, currency
      FROM `{HIST_TABLE}` {clause}
      ORDER BY id DESC
      LIMIT %s OFFSET %s
    """, tuple(params+[per, off]))
    return total, pages, page, rows


app = Flask(__name__)
PIPE: Optional[Pipeline]=None
METRICS=None
CHOICES={"cities":[], "boards":[], "transports":[], "rooms":[], "hotels":[]}
META={"table":"?","target":"?","log_target":USE_LOG_TARGET,"temporal":USE_TEMPORAL_SPLIT}

@app.route("/", methods=["GET"])
def home():
    return render_template_string(
        HTML_TPL,
        cities=CHOICES["cities"], boards=CHOICES["boards"], transports=CHOICES["transports"],
        rooms=CHOICES["rooms"], hotels=CHOICES["hotels"],
        metrics=METRICS, counts={"train":METRICS["n_train"],"test":METRICS["n_test"],"features":METRICS["n_features"]},
        prediction=None, currency=PIPE.currency, meta=META
    )

@app.route("/predict", methods=["POST"])
def predict():
    form = {
        "city": request.form.get("city"),
        "hotel": request.form.get("hotel"),
        "nights": int(request.form.get("nights")),
        "stars": int(request.form.get("stars")),
        "board": request.form.get("board"),
        "transport": request.form.get("transport"),
        "room": request.form.get("room"),
        "month": int(request.form.get("month"))
    }
    y_hat = PIPE.predict_one(form, USE_LOG_TARGET)
    # upis istorije
    try:
        insert_history_row(form, y_hat, PIPE.currency, METRICS, META)
    except Exception as e:
        print(f"[WARN] Insert history failed: {e}")

    return render_template_string(
        HTML_TPL,
        cities=CHOICES["cities"], boards=CHOICES["boards"], transports=CHOICES["transports"],
        rooms=CHOICES["rooms"], hotels=CHOICES["hotels"],
        metrics=METRICS, counts={"train":METRICS["n_train"],"test":METRICS["n_test"],"features":METRICS["n_features"]},
        prediction=y_hat, currency=PIPE.currency, meta=META
    )

@app.route("/history", methods=["GET"])
def history_page():
    # filteri / paginacija
    try:
        per = min(max(1, int(request.args.get("per") or HIST_DEFAULT_PER)), HIST_MAX_PER)
    except:
        per = HIST_DEFAULT_PER
    try:
        page = max(1, int(request.args.get("page") or "1"))
    except:
        page = 1
    total, pages, page, rows = query_history(request.args, page, per)

    # priprema UI podataka
    items = []
    for r in rows:
        ua = r.get("user_agent") or ""
        items.append({
            "id": r["id"],
            "created_at": r["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
            "ip": r.get("ip") or "",
            "user_agent_short": (ua[:60] + "…") if len(ua) > 60 else ua,
            "city": r.get("city") or "",
            "hotel": r.get("hotel") or "",
            "nights": r.get("nights") or 0,
            "stars": r.get("stars") or 0,
            "board": r.get("board") or "",
            "transport": r.get("transport") or "",
            "room": r.get("room") or "",
            "month": r.get("month") or 0,
            "prediction": float(r.get("prediction") or 0.0),
            "currency": r.get("currency") or "",
        })

    # liste za filtere (iz choices/enc)
    cities = CHOICES["cities"]
    boards = CHOICES["boards"]

    def pager_url(p):
        args = request.args.to_dict(flat=True)
        args["page"] = str(p)
        args["per"] = str(per)
        q = "&".join([f"{k}={v}" for k,v in args.items() if v!=""])
        return f"/history?{q}" if q else "/history"

    return render_template_string(
        HTML_HIST,
        items=items, total=total, pages=pages, page=page, per=per,
        cities=cities, boards=boards,
        flt={
            "city": request.args.get("city") or "",
            "board": request.args.get("board") or "",
            "month": int(request.args.get("month")) if request.args.get("month") and request.args.get("month").isdigit() else None
        },
        pager_url=pager_url
    )

@app.route("/history.csv", methods=["GET"])
def history_csv():
    try:
        per = min(int(request.args.get("limit") or 10000), 10000)
    except:
        per = 10000
    total, pages, page, rows = query_history(request.args, 1, per)
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id","created_at","ip","user_agent","city","hotel","nights","stars","board","transport","room","month","prediction","currency"])
    for r in rows:
        writer.writerow([
            r["id"],
            r["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
            r.get("ip") or "",
            r.get("user_agent") or "",
            r.get("city") or "",
            r.get("hotel") or "",
            r.get("nights") or 0,
            r.get("stars") or 0,
            r.get("board") or "",
            r.get("transport") or "",
            r.get("room") or "",
            r.get("month") or 0,
            float(r.get("prediction") or 0.0),
            r.get("currency") or "",
        ])
    csv_data = output.getvalue()
    output.close()
    return Response(
        csv_data,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=prediction_history.csv"}
    )

@app.route("/api/history", methods=["GET"])
def history_api():
    try:
        per = min(max(1, int(request.args.get("limit") or HIST_DEFAULT_PER)), HIST_MAX_PER)
    except:
        per = HIST_DEFAULT_PER
    try:
        page = max(1, int(request.args.get("page") or "1"))
    except:
        page = 1
    total, pages, page, rows = query_history(request.args, page, per)
    # JSON
    data = {
        "page": page,
        "pages": pages,
        "per": per,
        "total": total,
        "items": [{
            "id": r["id"],
            "created_at": r["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
            "ip": r.get("ip"),
            "user_agent": r.get("user_agent"),
            "city": r.get("city"),
            "hotel": r.get("hotel"),
            "nights": r.get("nights"),
            "stars": r.get("stars"),
            "board": r.get("board"),
            "transport": r.get("transport"),
            "room": r.get("room"),
            "month": r.get("month"),
            "prediction": float(r.get("prediction") or 0.0),
            "currency": r.get("currency"),
        } for r in rows]
    }
    return jsonify(data)


def main():
    try:
        cnx=connect_db(); db=cnx.database; cnx.close()
    except Error as e:
        print(f"[ERR] DB konekcija: {e}"); return

    # osiguraj tabelu istorije
    try:
        ensure_history_table()
    except Exception as e:
        print(f"[WARN] ensure_history_table: {e}")

    try:
        table,target=pick_table_and_target()
    except Exception as e:
        print(f"[ERR] {e}"); return

    rows=fetch_rows(table,target)
    if not rows:
        print("[ERR] Nema redova posle čitanja."); return

    set_seed(RANDOM_SEED)
    global PIPE, METRICS, CHOICES, META
    PIPE=Pipeline(table,target)
    METRICS, rows_train, rows_test = PIPE.fit(rows, log_target=USE_LOG_TARGET, temporal_split=USE_TEMPORAL_SPLIT)
    META={"table":table, "target":target, "log_target":USE_LOG_TARGET, "temporal":USE_TEMPORAL_SPLIT}

    # UI liste iz encoder vocaba (poklapa se sa bucket pravilima)
    CHOICES["cities"]     = PIPE.enc.vocabs.get("city", []) or sorted({r["city"] for r in rows})
    CHOICES["boards"]     = PIPE.enc.vocabs.get("board_cat", []) or ["BED_BREAKFAST","HALF_BOARD","FULL_BOARD","ALL_INCLUSIVE","ULTRA_AI","AI_PREMIUM","SELF_CATERING","ROOM_ONLY","BOARD_OTHER"]
    CHOICES["transports"] = PIPE.enc.vocabs.get("transport_mode", []) or sorted({(r.get("transport_mode") or "other") for r in rows})
    CHOICES["rooms"]      = PIPE.enc.vocabs.get("room_group", []) or sorted({(r.get("room_group") or "ROOM_OTHER") for r in rows})

    # hotels — top po učestalosti (bez OTHER_HOTEL)
    if "hotel_title" in PIPE.enc.vocabs:
        hotels_vocab = [h for h in PIPE.enc.vocabs["hotel_title"] if h!="OTHER_HOTEL"]
        CHOICES["hotels"] = hotels_vocab[:150]
    else:
        cnt = Counter([r["title"] for r in rows_train])
        CHOICES["hotels"] = [h for (h,_) in cnt.most_common(150)]

    print("[EVAL] MAE={mae:.2f} RMSE={rmse:.2f} MAPE={mape:.2f}% R²={r2:.4f} | feat={n_features} train={n_train} test={n_test}".format(**METRICS))
    print(f"[OK] Valuta: {'EUR' if 'eur' in target.lower() else 'RSD'} | Log-target: {USE_LOG_TARGET} | Split: {'temporal' if USE_TEMPORAL_SPLIT else 'random'}")

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

if __name__=="__main__":
    main()
