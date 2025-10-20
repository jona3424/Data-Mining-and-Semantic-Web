
import os
import re
import json
import argparse
import random
from glob import glob
from datetime import datetime
from typing import Any, Dict, Optional, Iterable, Tuple, List, Callable

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

DEFAULT_DB_URL = "mysql+pymysql://root:@localhost:3306/travel_etl?charset=utf8mb4"
DEFAULT_FOLDER = "winter"
ASSUME_EUR_WHEN_CURRENCY_MISSING = True

KOLASIN_FILE_BASENAME = "kolasin_outputs_winter.json"
KOLASIN_MAX_ROWS = 10_000
MARIBORSKO_FILE_BASENAME = "mariborsko_outputs_winter.json"
ROGLA_FILE_BASENAME = "rogla_outputs_winter.json"
ROGLA_MAX_ROWS = 1_000

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


import math

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
        s = str(x).strip()
        s = s.replace(" ", "").replace("\xa0", "").replace(",", ".")
        s = re.sub(r"[^\d.\-eE]", "", s)
        return float(s)
    except Exception:
        return None

def parse_price_text(price_text: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    if not price_text:
        return None, None
    txt = str(price_text)
    cur = None
    if re.search(r"\bEUR\b|€", txt, flags=re.I):
        cur = "EUR"
    elif re.search(r"\bRSD\b|дин", txt, flags=re.I):
        cur = "RSD"
    val = to_float(txt)
    return val, cur

def parse_date(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.strip()
    try:
        if re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", s):
            return datetime.strptime(s, "%d.%m.%Y").strftime("%Y-%m-%d")
        return datetime.fromisoformat(s).strftime("%Y-%m-%d")
    except Exception:
        return None

def split_location(loc: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not loc: return None, None
    parts = [p.strip() for p in loc.split("-")]
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return None, None

def stars_from_raw(stars_raw: Any) -> Optional[int]:
    v = to_int(stars_raw)
    if v is None: return None
    return (v // 2) if v > 5 else v

def flag_all_inclusive(board: Optional[str]) -> Optional[int]:
    if board is None: return None
    return 1 if "ALL INCLUSIVE" in (board or "").upper() else 0

def make_ident_key(dep: Optional[str], nights: Optional[int],
                   room: Optional[str], board: Optional[str]) -> str:
    dep_key = dep or "NA"
    n_key   = str(nights if nights is not None else -1)
    r_key   = (norm_ws(room) or "").lower()
    b_key   = (norm_ws(board) or "").lower()
    return f"{dep_key}|{n_key}|{r_key}|{b_key}"


UNIT_MAP = {
    "po sobi": "room",
    "per room": "room",
    "po smještaju": "room",
    "po smjestaju": "room",
    "po objektu": "room",
    "po osobi": "person",
    "per person": "person",
}

def norm_unit(u: Optional[str]) -> Optional[str]:
    if not u: return None
    s = norm_ws(u).lower()
    return UNIT_MAP.get(s, s)

def _find_num(txt: Optional[str]) -> Optional[int]:
    if not txt: return None
    m = re.search(r"(\d+)", txt)
    return int(m.group(1)) if m else None

def pax_from_capacity(cap: Optional[str]) -> Optional[int]:
    if not cap:
        return None
    s = cap.lower()
    rng = re.search(r"(\d+)\s*-\s*(\d+)", s)
    if rng:
        try:
            a, b = int(rng.group(1)), int(rng.group(2))
            return max(a, b)
        except Exception:
            pass
    single = re.search(r"(\d+)\s*osob", s)
    if single:
        try:
            return int(single.group(1))
        except Exception:
            pass
    n = _find_num(s)
    return n

def pax_from_text(unit: Optional[str], price_note: Optional[str], capacity_text: Optional[str]) -> int:
    p = pax_from_capacity(capacity_text)
    if p and p > 0:
        return p
    nu = norm_unit(unit)
    if nu == "person":
        return 1
    n = _find_num(unit) or _find_num(price_note)
    if n and n > 0:
        return n
    return 2 if (nu == "room" or nu is None) else 2

def build_room_type(room_name: Optional[str],
                    unit: Optional[str],
                    pax: int,
                    room_id: Optional[str],
                    rate_id: Optional[str]) -> str:
    base = f"room={norm_ws(room_name) or 'N/A'} | unit={(norm_unit(unit) or 'unknown')};pax{pax}"
    tag_room = f" #room={room_id}" if room_id else ""
    tag_rate = f" #rate={rate_id}" if rate_id else ""
    return (base + tag_room + tag_rate).strip()


def _hotels_list_from_night_val(val: Any) -> List[Dict[str, Any]]:
    if isinstance(val, list):
        return [h for h in val if isinstance(h, dict)]
    if isinstance(val, dict):
        hotels = val.get("hotels")
        if isinstance(hotels, list):
            return [h for h in hotels if isinstance(h, dict)]
    return []

def _offers_map_from_night_val(val: Any) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    if isinstance(val, dict) and isinstance(val.get("offers_by_hotel"), dict):
        return val["offers_by_hotel"]
    return None

def _currency_to_targets(currency: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if currency:
        cu = currency.upper()
        if cu == "EUR":
            return "EUR", None
        if cu == "RSD":
            return None, "RSD"
    if ASSUME_EUR_WHEN_CURRENCY_MISSING:
        return "EUR", None
    return None, None

def _extract_price_fields(price_num: Any,
                          currency: Optional[str],
                          price_text: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    eur_target, rsd_target = _currency_to_targets(currency)
    if price_num is not None:
        val = to_float(price_num)
        if eur_target:
            return val, None
        if rsd_target:
            return None, val
        if ASSUME_EUR_WHEN_CURRENCY_MISSING:
            return val, None
        return None, None
    if price_text:
        val_txt, cur_txt = parse_price_text(price_text)
        if val_txt is not None:
            eur_target2, rsd_target2 = _currency_to_targets(cur_txt)
            if eur_target2:
                return val_txt, None
            if rsd_target2:
                return None, val_txt
            if ASSUME_EUR_WHEN_CURRENCY_MISSING:
                return val_txt, None
    return None, None


def iter_offers_winter_tree(
    tree: Dict[str, Any],
    location_url: Optional[str]=None,
    date_filter: Optional[Callable[[str], bool]] = None
) -> Iterable[Dict[str, Any]]:
    by_date = tree or {}
    site = "kontiki"
    url  = location_url or ""
    for dep_date, nights_dict in by_date.items():
        dep_iso = parse_date(dep_date)
        if dep_iso is None or not isinstance(nights_dict, dict):
            continue
        if date_filter and not date_filter(dep_iso):
            continue
        for nights_str, val in nights_dict.items():
            if not re.fullmatch(r"\d+", str(nights_str)):
                continue
            nights = to_int(nights_str)
            n_fac = int(nights or 1)
            hotels = _hotels_list_from_night_val(val)
            offers_map = _offers_map_from_night_val(val)

            if not offers_map:
                for h in hotels:
                    title = norm_ws(h.get("name") or "")
                    city  = norm_ws(h.get("city"))
                    stars_raw = h.get("stars")
                    unit = h.get("unit")
                    price_note = h.get("price_note")
                    pax = pax_from_text(unit, price_note, h.get("capacity_text"))
                    val_eur, val_rsd = _extract_price_fields(
                        h.get("price_num"),
                        (h.get("currency") or "").upper() if h.get("currency") else None,
                        h.get("price_text"),
                    )
                    if val_eur is None and val_rsd is None:
                        continue
                    nu = norm_unit(unit)
                    unit_is_person = (nu == "person")
                    price_eur = price_rsd = None
                    if val_eur is not None:
                        total = float(val_eur) * n_fac
                        if unit_is_person:
                            total *= 2
                        price_eur = to_float(total)
                    elif val_rsd is not None:
                        total = float(val_rsd) * n_fac
                        if unit_is_person:
                            total *= 2
                        price_rsd = to_float(total)
                    room_type = build_room_type(h.get("room_name") or h.get("name"), unit, pax, None, None)
                    board = norm_ws(h.get("board"))
                    yield {
                        "site": site, "url": url, "title": title, "location": city, "stars_raw": stars_raw,
                        "departure_date_raw": dep_iso, "nights": nights,
                        "rooms": [{
                            "room_type": room_type, "board": board,
                            "old_price_eur": None, "old_price_rsd": None,
                            "price_eur": price_eur, "price_rsd": price_rsd,
                        }],
                    }
                continue

            hotels_meta = { (h.get("name") or "").strip(): h for h in hotels if isinstance(h, dict) }
            for hotel_name, offers in offers_map.items():
                meta = hotels_meta.get(hotel_name, {})
                title = norm_ws(hotel_name) or norm_ws(meta.get("name")) or ""
                city  = norm_ws(meta.get("city"))
                stars_raw = meta.get("stars")
                if not isinstance(offers, list):
                    continue
                for off in offers:
                    if not isinstance(off, dict):
                        continue
                    board = norm_ws(off.get("board"))
                    capacity_text = off.get("capacity_text")
                    room_name = off.get("room_name")
                    nights_off = to_int(off.get("nights")) or nights
                    price_total = off.get("price_total")
                    price_per_night = off.get("price_per_night")
                    currency = (off.get("currency") or "").upper() if off.get("currency") else None
                    if price_total is None and (price_per_night is not None) and (nights_off is not None):
                        try:
                            price_total = float(price_per_night) * int(nights_off)
                        except Exception:
                            price_total = None
                    if price_total is None:
                        raw = off.get("raw") or {}
                        val_txt, cur_txt = parse_price_text((raw.get("total_text") or "") + " " + (raw.get("price_text") or ""))
                        if val_txt is not None and nights_off:
                            price_total = float(val_txt) * int(nights_off)
                            currency = currency or cur_txt
                    if price_total is None:
                        continue
                    raw = off.get("raw") or {}
                    unit = raw.get("price_note") or off.get("unit")
                    nu = norm_unit(unit)
                    if nu == "person":
                        price_total = float(price_total) * 2
                    room_id = (raw.get("room_id") if isinstance(raw, dict) else None) or off.get("room_id")
                    rate_id = (raw.get("rate_id") if isinstance(raw, dict) else None) or off.get("rate_id")
                    pax = pax_from_text(unit, raw.get("price_note") if isinstance(raw, dict) else None, capacity_text)
                    eur_target, rsd_target = _currency_to_targets(currency)
                    price_eur = to_float(price_total) if eur_target else None
                    price_rsd = to_float(price_total) if rsd_target else None
                    if price_eur is None and price_rsd is None and ASSUME_EUR_WHEN_CURRENCY_MISSING:
                        price_eur = to_float(price_total)
                    room_type = build_room_type(room_name, unit, pax, room_id, rate_id)
                    yield {
                        "site": "kontiki", "url": location_url or "", "title": title, "location": city,
                        "stars_raw": stars_raw, "departure_date_raw": dep_iso, "nights": nights_off,
                        "rooms": [{
                            "room_type": room_type, "board": board,
                            "old_price_eur": None, "old_price_rsd": None,
                            "price_eur": price_eur, "price_rsd": price_rsd,
                        }],
                    }

def iter_offers_from_file(
    path: str,
    date_filter: Optional[Callable[[str], bool]] = None
) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if (isinstance(data, dict) and "rooms" in data) or isinstance(data, list):
        if isinstance(data, list):
            for o in data:
                if isinstance(o, dict) and "rooms" in o:
                    yield o
        elif isinstance(data, dict):
            yield data
        return
    if isinstance(data, dict) and "by_date" in data:
        yield from iter_offers_winter_tree(data.get("by_date") or {}, data.get("location_url"), date_filter)
        return
    if isinstance(data, dict) and len(data) == 1:
        only_val = next(iter(data.values()))
        if isinstance(only_val, dict) and "by_date" in only_val:
            yield from iter_offers_winter_tree(only_val.get("by_date") or {}, only_val.get("location_url"), date_filter)
            return
    if isinstance(data, dict) and data:
        looks_like_date = any(isinstance(k, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", k)
                              for k in list(data.keys())[:10])
        if looks_like_date:
            yield from iter_offers_winter_tree(data, data.get("location_url"), date_filter)
            return
    return


DEDUP_FIELDS = [
    "site", "url", "ident_key", "title", "location_text", "country", "city", "stars",
    "departure_date", "nights", "room_type", "board", "is_all_inclusive",
    "old_price_eur", "old_price_rsd", "price_eur", "price_rsd"
]

def params_from_offer(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    site   = obj.get("site") or "unknown"
    url    = obj.get("url") or ""
    title  = norm_ws(obj.get("title") or "")
    loc    = norm_ws(obj.get("location"))
    stars  = stars_from_raw(obj.get("stars_raw"))
    dep    = parse_date(obj.get("departure_date_raw"))
    nights = to_int(obj.get("nights"))
    country, city = split_location(loc)
    if not city and loc:
        city = loc

    rows: List[Dict[str, Any]] = []
    rooms = obj.get("rooms") or []
    for r in rooms:
        room  = norm_ws(r.get("room_type"))
        board = norm_ws(r.get("board"))
        old_price_eur = to_float(r.get("old_price_eur"))
        old_price_rsd = to_float(r.get("old_price_rsd"))
        price_eur     = to_float(r.get("price_eur"))
        price_rsd     = to_float(r.get("price_rsd"))

        if all(v is None for v in (old_price_eur, old_price_rsd, price_eur, price_rsd)):
            continue

        ident = make_ident_key(dep, nights, room, board)
        rows.append(dict(
            site=site, url=url, ident_key=ident, title=title,
            location_text=loc, country=country, city=city, stars=stars,
            departure_date=dep, nights=nights,
            room_type=room, board=board, is_all_inclusive=flag_all_inclusive(board),
            old_price_eur=old_price_eur, old_price_rsd=old_price_rsd,
            price_eur=price_eur, price_rsd=price_rsd
        ))
    return rows

def insert_rows(engine: Engine, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    with engine.begin() as conn:
        for p in rows:
            conn.execute(INSERT_UPSERT, p)
    return len(rows)


def load_single_file(engine: Engine, path: str, limit: int, sample_seed: Optional[int]) -> None:
    if sample_seed is not None:
        random.seed(sample_seed)

    seen = set()
    sample: List[Dict[str, Any]] = []
    emitted_offers = 0
    uniq_count = 0

    for offer in iter_offers_from_file(path):
        emitted_offers += 1
        for p in params_from_offer(offer):
            key = tuple(p[f] for f in DEDUP_FIELDS)
            if key in seen:
                continue
            seen.add(key)
            if uniq_count < limit:
                sample.append(p)
            else:
                j = random.randint(0, uniq_count)
                if j < limit:
                    sample[j] = p
            uniq_count += 1

    random.shuffle(sample)
    n = insert_rows(engine, sample)
    base = os.path.basename(path)
    print(f"[FILE] {base} -> {n} upisanih soba (random max {limit}, {len(seen)} jedinstvenih kandidata, iz {emitted_offers} hotelskih zapisa)")

def load_folder(engine: Engine, folder: str, sample_seed: Optional[int]=None) -> None:
    if sample_seed is not None:
        random.seed(sample_seed)

    files = sorted(glob(os.path.join(folder, "*.json")) + glob(os.path.join(folder, "*.ndjson")))
    total = len(files)
    if total == 0:
        print(f"[WARN] Nema .json fajlova u: {folder}")
        return

    for i, fp in enumerate(files, 1):
        base = os.path.basename(fp)
        try:
            if base == MARIBORSKO_FILE_BASENAME:
                def only_dec_2025(dep_iso: str) -> bool:
                    return dep_iso.startswith("2025-12-")
                picked: List[Dict[str, Any]] = []
                emitted_offers = 0
                for offer in iter_offers_from_file(fp, date_filter=only_dec_2025):
                    emitted_offers += 1
                    picked.extend(params_from_offer(offer))
                n = insert_rows(engine, picked)
                print(f"[{i}/{total}] {base} -> {n} upisanih soba (samo decembar 2025; iz {emitted_offers} hotelskih zapisa)")
                continue

            if base == KOLASIN_FILE_BASENAME:
                seen = set()
                picked: List[Dict[str, Any]] = []
                emitted_offers = 0
                for offer in iter_offers_from_file(fp):
                    emitted_offers += 1
                    for p in params_from_offer(offer):
                        key = tuple(p[f] for f in DEDUP_FIELDS)
                        if key in seen:
                            continue
                        seen.add(key)
                        picked.append(p)
                        if len(picked) >= KOLASIN_MAX_ROWS:
                            break
                    if len(picked) >= KOLASIN_MAX_ROWS:
                        break
                n = insert_rows(engine, picked)
                print(f"[{i}/{total}] {base} -> {n} upisanih soba (limit {KOLASIN_MAX_ROWS}, {len(seen)} jedinstvenih kandidata iz {emitted_offers} hotelskih zapisa)")
                continue

            if base == ROGLA_FILE_BASENAME:
                seen = set()
                sample: List[Dict[str, Any]] = []
                emitted_offers = 0
                uniq_count = 0
                for offer in iter_offers_from_file(fp):
                    emitted_offers += 1
                    for p in params_from_offer(offer):
                        key = tuple(p[f] for f in DEDUP_FIELDS)
                        if key in seen:
                            continue
                        seen.add(key)
                        if uniq_count < ROGLA_MAX_ROWS:
                            sample.append(p)
                        else:
                            j = random.randint(0, uniq_count)
                            if j < ROGLA_MAX_ROWS:
                                sample[j] = p
                        uniq_count += 1
                random.shuffle(sample)
                n = insert_rows(engine, sample)
                print(f"[{i}/{total}] {base} -> {n} upisanih soba (random max {ROGLA_MAX_ROWS}, {len(seen)} jedinstvenih kandidata iz {emitted_offers} hotelskih zapisa)")
                continue

            batch: List[Dict[str, Any]] = []
            emitted_offers = 0
            for offer in iter_offers_from_file(fp):
                emitted_offers += 1
                batch.extend(params_from_offer(offer))
            n = insert_rows(engine, batch)
            print(f"[{i}/{total}] {base} -> {n} upisanih soba (iz {emitted_offers} hotelskih zapisa)")

        except Exception as e:
            print(f"[ERR] {fp}: {e}")


def parse_args():
    p = argparse.ArgumentParser(description="Loader za winter + legacy JSON formate -> arrangements (skala noći & 2 pax; specijalni filteri; single-file sampling)")
    p.add_argument("--db", dest="db_url", default=DEFAULT_DB_URL, help="SQLAlchemy DB URL")
    p.add_argument("--folder", default=DEFAULT_FOLDER, help="Folder sa JSON fajlovima (ignoriše se ako je zadat --file)")
    p.add_argument("--file", help="PUTANJA do jednog JSON fajla koji će biti obrađen samostalno")
    p.add_argument("--limit", type=int, default=1000, help="Max broj jedinstvenih redova kada je zadat --file (default 1000)")
    p.add_argument("--sample-seed", type=int, default=None, help="Seed za nasumični izbor (reproduktivnost)")
    return p.parse_args()

def main():
    args = parse_args()
    engine = create_engine(args.db_url, pool_pre_ping=True, future=True)
    if args.file:
        load_single_file(engine, args.file, args.limit, args.sample_seed)
    else:
        load_folder(engine, args.folder, sample_seed=args.sample_seed)

if __name__ == "__main__":
    main()
