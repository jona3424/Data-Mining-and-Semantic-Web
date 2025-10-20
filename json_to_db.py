import os
import re
import json
from glob import glob
from datetime import datetime
from typing import Any, Dict, Optional, Iterable

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ====== PODESI ======
MYSQL_URL   = "mysql+pymysql://root:@localhost:3306/travel_etl?charset=utf8mb4"
OUTPUT_DIR  = "outputs"  
# ====================

INSERT_UPSERT = text("""
INSERT INTO arrangements
(site, url, ident_key, title, location_text, country, city, stars,
 departure_date, nights, room_type, board, is_all_inclusive,
 old_price_eur, old_price_rsd, price_eur, price_rsd)
VALUES
(:site, :url, :ident_key, :title, :location_text, :country, :city, :stars,
 :departure_date, :nights, :room_type, :board, :is_all_inclusive,
 :old_price_eur, :old_price_rsd, :price_eur, :price_rsd)
ON DUPLICATE KEY UPDATE
  title            = VALUES(title),
  location_text    = VALUES(location_text),
  country          = IFNULL(VALUES(country), country),
  city             = IFNULL(VALUES(city), city),
  stars            = IFNULL(VALUES(stars), stars),
  departure_date   = IFNULL(VALUES(departure_date), departure_date),
  nights           = IFNULL(VALUES(nights), nights),
  room_type        = IFNULL(VALUES(room_type), room_type),
  board            = IFNULL(VALUES(board), board),
  is_all_inclusive = VALUES(is_all_inclusive),
  old_price_eur    = VALUES(old_price_eur),
  old_price_rsd    = VALUES(old_price_rsd),
  price_eur        = VALUES(price_eur),
  price_rsd        = VALUES(price_rsd)
""")

def norm_ws(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    return re.sub(r"\s+", " ", s).strip()

def to_int(x: Any) -> Optional[int]:
    try:
        if x is None: return None
        return int(str(x).strip())
    except Exception:
        return None

def to_float(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        s = str(x).strip().replace(" ", "").replace(",", "")
        return float(s)
    except Exception:
        return None

def parse_date(s: Optional[str]) -> Optional[str]:
    """Podržava 'dd.mm.yyyy' i ISO 'yyyy-mm-dd'."""
    if not s: return None
    s = s.strip()
    try:
        if re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", s):
            return datetime.strptime(s, "%d.%m.%Y").strftime("%Y-%m-%d")
        return datetime.fromisoformat(s).strftime("%Y-%m-%d")
    except Exception:
        return None

def split_location(loc: Optional[str]) -> (Optional[str], Optional[str]):
    if not loc: return None, None
    parts = [p.strip() for p in loc.split("-")]
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return None, None

def stars_from_raw(stars_raw: Any) -> Optional[int]:
    v = to_int(stars_raw)
    return None if v is None else v // 2  # tvoje pravilo

def flag_all_inclusive(board: Optional[str]) -> Optional[int]:
    if board is None: return None
    return 1 if "ALL INCLUSIVE" in board.upper() else 0

def make_ident_key(dep: Optional[str], nights: Optional[int],
                   room: Optional[str], board: Optional[str]) -> str:
    dep_key = dep or "NA"
    n_key   = str(nights if nights is not None else -1)
    r_key   = (norm_ws(room) or "").lower()
    b_key   = (norm_ws(board) or "").lower()
    return f"{dep_key}|{n_key}|{r_key}|{b_key}"

def iter_offers_from_file(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict) and "rooms" in obj:
                yield obj
    elif isinstance(data, dict) and "rooms" in data:
        yield data

def load_offer(engine: Engine, obj: Dict[str, Any]) -> None:
    site   = obj.get("site") or "unknown"
    url    = obj.get("url") or ""
    title  = norm_ws(obj.get("title") or "")
    loc    = norm_ws(obj.get("location"))
    stars  = stars_from_raw(obj.get("stars_raw"))
    dep    = parse_date(obj.get("departure_date_raw"))
    nights = to_int(obj.get("nights"))
    country, city = split_location(loc)

    rooms = obj.get("rooms") or []
    if not rooms:
        # upiši jedan red i ako nema sobu ili cijenu
        ident = make_ident_key(dep, nights, None, None)
        params = dict(
            site=site, url=url, ident_key=ident, title=title,
            location_text=loc, country=country, city=city, stars=stars,
            departure_date=dep, nights=nights,
            room_type=None, board=None, is_all_inclusive=None,
            old_price_eur=None, old_price_rsd=None,
            price_eur=None, price_rsd=None
        )
        with engine.begin() as conn:
            conn.execute(INSERT_UPSERT, params)
        return

    for r in rooms:
        room  = norm_ws(r.get("room_type"))
        board = norm_ws(r.get("board"))
        ident = make_ident_key(dep, nights, room, board)
        params = dict(
            site=site, url=url, ident_key=ident, title=title,
            location_text=loc, country=country, city=city, stars=stars,
            departure_date=dep, nights=nights,
            room_type=room, board=board, is_all_inclusive=flag_all_inclusive(board),
            old_price_eur=to_float(r.get("old_price_eur")),
            old_price_rsd=to_float(r.get("old_price_rsd")),
            price_eur=to_float(r.get("price_eur")),
            price_rsd=to_float(r.get("price_rsd"))
        )
        with engine.begin() as conn:
            conn.execute(INSERT_UPSERT, params)

def load_outputs_folder(engine: Engine, folder: str) -> None:
    files = sorted(glob(os.path.join(folder, "*.json")))
    for i, fp in enumerate(files, 1):
        try:
            for offer in iter_offers_from_file(fp):
                load_offer(engine, offer)
            # print(f"[{i}/{len(files)}] {os.path.basename(fp)} OK")
        except Exception as e:
            print(f"[ERR] {fp}: {e}")

def main():
    engine = create_engine(MYSQL_URL, pool_pre_ping=True, future=True)
    # kod za samo jedan file: load_file(engine, "outputs/offers_kontiki_malta.json")
    load_outputs_folder(engine, OUTPUT_DIR)

if __name__ == "__main__":
    main()
