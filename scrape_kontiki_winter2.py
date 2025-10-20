
import asyncio, json, argparse, re
from datetime import date, timedelta
from typing import Dict, Any, List, Optional
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

BASE = "https://kontiki.rs"
START = f"{BASE}/sr/skijanje-i-zimovanje"

IFRAMES = ",".join([
    "iframe#cruisepool_iframe",
    "iframe[name='cruisepool_iframe']",
    "iframe[src*='phobs']",
    "iframe[src*='engine']",
    "iframe[src*='frontoffice']",
])

LIST_ANY = ", ".join([
    ".property-show-box", ".property-show-table",
    ".property, .property-item, .hotel",
    "#SearchResult"
])

DETAIL_ANY = ", ".join([
    "#all_units .unit-definition-box",
    ".room-rates-table",
    "form.rate-plan-holder",
    ".rates-table",
    ".unit-definition-box"
])

SR_MONTHS = {1:"Januar",2:"Februar",3:"Mart",4:"April",5:"Maj",6:"Jun",7:"Jul",8:"Avgust",9:"Septembar",10:"Oktobar",11:"Novembar",12:"Decembar"}

NO_ROOMS_PAT = re.compile(
    r"(nema\s+soba|nema\s+raspoloživih|no\s+available\s+rooms|there\s+are\s+no\s+available\s+rooms)",
    re.I
)

def ymd(d: date) -> str: return f"{d.year:04d}-{d.month:02d}-{d.day:02d}"
def daterange_inclusive(a: date, b: date):
    if b < a: a, b = b, a
    cur = a
    while cur <= b:
        yield cur
        cur += timedelta(days=1)

def normalize_loc(u: str) -> str:
    return u if u.startswith("http") else f"{BASE}/sr/location/{u.strip('/').split('/')[-1]}"

# iframe helpers
async def _current_booking_frame(page):
    # Pokusaj da nadjes booking frame direktno pa iz njega izvadi child-ove
    host = await page.wait_for_selector(IFRAMES, timeout=9000) 
    outer = await host.content_frame()
    if not outer:
        raise RuntimeError("Nema outer content_frame()")

    async def is_booking(f):
        try:
            return bool(await f.locator("#SearchResult, #quick_check_in_day, #quick_nights").count())
        except:
            return False

    if await is_booking(outer):
        return outer

    # Potrazi child-iframeove
    try:
        kids = outer.locator("iframe")
        for i in range(await kids.count()):
            cf = await kids.nth(i).content_frame()
            if cf and await is_booking(cf):
                return cf
    except:
        pass
    return outer

async def _ensure_availability_tab(frame):
    try:
        if not await frame.locator("#quick_check_in_day, #quick_nights").count():
            t = frame.locator("a,button").filter(has_text=re.compile("Raspoloživost|Availability|RASPOLOŽIVOST|Pretraži|Traži|Search", re.I))
            if await t.count():
                await t.first.click()
                await frame.wait_for_timeout(120)
    except:
        pass

NIGHTS = ["#quick_nights","select[name*='night']","select[id*='night']"]
IN_DAY = ["#quick_check_in_day","select[id*='check_in_day']","select[name*='check_in_day']","select[name*='checkin_day']","select[id*='arrival_day']","select[name*='arrival_day']"]
IN_MON = ["#quick_check_in_month","select[id*='check_in_month']","select[name*='check_in_month']","select[name*='checkin_month']","select[id*='arrival_month']","select[name*='arrival_month']"]
IN_YEA = ["#quick_check_in_year","select[id*='check_in_year']","select[name*='check_in_year']","select[name*='checkin_year']","select[id*='arrival_year']","select[name*='arrival_year']"]
OU_DAY = ["#quick_check_out_day","select[id*='check_out_day']","select[name*='check_out_day']","select[name*='checkout_day']","select[id*='departure_day']","select[name*='departure_day']"]
OU_MON = ["#quick_check_out_month","select[id*='check_out_month']","select[name*='check_out_month']","select[id*='departure_month']","select[name*='departure_month']"]
OU_YEA = ["#quick_check_out_year","select[id*='check_out_year']","select[name*='check_out_year']","select[name*='checkout_year']","select[id*='departure_year']","select[name*='departure_year']"]

async def _first(frame, cands):
    for c in cands:
        try:
            if await frame.locator(c).count():
                return c
        except:
            pass
    return None

async def _sel(frame, sel, *, value=None, label=None):
    try:
        if label is not None:
            await frame.select_option(sel, label=str(label)); return True
    except: pass
    try:
        if value is not None:
            await frame.select_option(sel, value=str(value)); return True
    except: pass
    return False

async def _set_dates_and_search(frame, checkin: date, nights: int):
    checkout = checkin + timedelta(days=1*nights)
    await _ensure_availability_tab(frame)

    ns = await _first(frame, NIGHTS)
    if ns:
        await frame.locator(ns).wait_for(state="attached", timeout=6000)
        await _sel(frame, ns, value=str(nights), label=str(nights))
    else:
        ids, ims, iys = await _first(frame, IN_DAY), await _first(frame, IN_MON), await _first(frame, IN_YEA)
        ods, oms, oys = await _first(frame, OU_DAY), await _first(frame, OU_MON), await _first(frame, OU_YEA)
        if ids and ims and iys and ods and oms and oys:
            await _sel(frame, ids, value=f"{checkin.day:02d}", label=f"{checkin.day:02d}")
            await _sel(frame, ims, label=SR_MONTHS[checkin.month])
            await _sel(frame, iys, value=str(checkin.year), label=f"{checkin.year}.")
            await _sel(frame, ods, value=f"{checkout.day:02d}", label=f"{checkout.day:02d}")
            await _sel(frame, oms, label=SR_MONTHS[checkout.month])
            await _sel(frame, oys, value=str(checkout.year), label=f"{checkout.year}.")
        else:
            ci = frame.locator("input[name*='check_in'], input#check_in, input[name*='CheckIn']")
            co = frame.locator("input[name*='check_out'], input#check_out, input[name*='CheckOut']")
            if await ci.count(): await ci.first.fill(f"{checkin.day:02d}.{checkin.month:02d}.{checkin.year}")
            if await co.count(): await co.first.fill(f"{checkout.day:02d}.{checkout.month:02d}.{checkout.year}")

    # Triger pretrage
    for sel in [
        "input[type='submit']",
        "button[type='submit']",
        "a:has-text('Pretraži')","a:has-text('Traži')","a:has-text('Search')","a:has-text('Izmeni')"
    ]:
        if await frame.locator(sel).count():
            await frame.locator(sel).first.click()
            break

    await frame.wait_for_selector(LIST_ANY, timeout=9000, state="attached")  # kraći timeout

# parsiranje liste hotela
async def _parse_hotels(frame) -> List[Dict[str, Any]]:
    js = r"""
(() => {
  const out = [];
  const uniq = new Set();
  const blocks = document.querySelectorAll(".property-show-box, .property-show-table, .property, .property-item, .hotel");
  const txt = n => n ? n.textContent.replace(/\s+/g,' ').trim() : null;
  blocks.forEach(b => {
    const name = txt(b.querySelector(".property-name, .property-title, .hotel-name, h3, h4"));
    if (!name) return;
    const city = txt(b.querySelector(".city-span, .city, .region"));
    const stars = (b.querySelectorAll(".property-category b")||[]).length || null;
    const key = (name+"|"+(city||"")).toLowerCase();
    if (uniq.has(key)) return;
    uniq.add(key);
    out.push({name, city, stars});
  });
  return out;
})();
"""
    return await frame.evaluate(js)

async def _row_has_choose_or_available(row) -> bool:
    # dugmad za izbor
    if await row.locator(
        "input[type='submit'][value*='Izaberi'], input[type='submit'][value*='Odaberi'], "
        ".button:has-text('IZABERI'), .button:has-text('ODABERI')"
    ).count():
        return True
    # ako se eksplicitno kaže da nema soba onda preskacemo
    txt = (await row.inner_text()).strip().lower()
    if NO_ROOMS_PAT.search(txt):
        return False
    # ako postoji link/forma ka detalju
    if await row.locator("form[action*='book'], a[href*='hotel'], .property-name, .property-title, .hotel-name").count():
        return True
    return False

async def _open_hotel_detail(page, frame, row):
    btn = row.locator(
        "input[name='book'][type='submit'], "
        "input[type='submit'][value*='Izaberi'], input[type='submit'][value*='Odaberi'], "
        ".button:has-text('IZABERI'), .button:has-text('ODABERI')"
    )

    # a) JS click 
    opened = False
    try:
        if await btn.count():
            el = btn.first
            await el.evaluate("b => b.click()")
            opened = True
        else:
            # probaj click na naslov
            await row.locator(".property-name, .property-title, .hotel-name, h3, h4").first.evaluate("e=>e.click()")
            opened = True
    except:
        pass

    if not opened:
        # b) submit forme
        try:
            await row.locator("form").first.evaluate("f => f.submit()")
            opened = True
        except:
            # c) force click
            if await btn.count():
                await btn.first.click(force=True)
                opened = True
            else:
                await row.locator(".property-name, .property-title, .hotel-name, h3, h4").first.click(force=True)
                opened = True

    # novi frame nakon navigacije
    new_frame = await _current_booking_frame(page)
    await new_frame.wait_for_selector(DETAIL_ANY, timeout=7000, state="attached")
    return new_frame

_money_re = re.compile(r'(\d{1,3}(?:[.\s]\d{3})*(?:,\d+)?|\d+(?:[.,]\d+)?)')
def _parse_money(text: Optional[str]):
    if not text: return None, None
    cur = None
    mcur = re.search(r'(EUR|RSD|€|дин)', text, re.I)
    if mcur: cur = mcur.group(1).upper().replace('€','EUR')
    m = _money_re.search(text.replace('\xa0',' ').replace(' ',' '))
    if not m: return None, cur
    val = m.group(1).replace('.','').replace(' ','').replace(',','.')
    try: return float(val), cur
    except: return None, cur

async def _scrape_units(frame, nights: int) -> List[Dict[str, Any]]:
    await frame.wait_for_selector(DETAIL_ANY, timeout=7000, state="attached")
    js = r"""
(() => {
  const out = [];
  const txt = n => n ? n.textContent.replace(/\s+/g,' ').trim() : null;
  const container = document.querySelector("#all_units") || document;
  const units = container.querySelectorAll(".unit-definition-box");
  const roots = units.length ? units : container.querySelectorAll(".room-rates-table, .rates-table");

  (roots.length ? roots : [container]).forEach(u => {
    const unit = u.closest(".unit-definition-box") || u;
    const roomName = txt(unit.querySelector(".room-name-box .room-name")) || txt(unit.querySelector(".room-name, .rate-name, .room-title, h3, h4"));
    const capacity = txt(unit.querySelector(".column-capacity"));
    const rows = unit.querySelectorAll(".rates-table tr, .room-rates-table tr");
    rows.forEach(r => {
      const boardTxt = txt(r.querySelector(".policy-meal")) || txt(r.querySelector(".rateplan-name, .rateplan-name-link"));
      const priceText = txt(r.querySelector(".price-value"));
      const priceNote = txt(r.querySelector(".price-desc"));
      const holder = r.closest("form.rate-plan-holder");
      let totalText = null, roomId=null, rateId=null;
      if (holder) {
        const d = holder.querySelector(".book-desc"); totalText = txt(d);
        const inRoom = holder.querySelector("input[name='room']"); roomId = inRoom ? inRoom.value : null;
        const inRate = holder.querySelector("input[name='rate']"); rateId = inRate ? inRate.value : null;
      }
      out.push({roomName, capacity, boardTxt, priceText, priceNote, totalText, roomId, rateId});
    });
  });
  return out;
})();
"""
    raw = await frame.evaluate(js)
    offers = []
    for r in raw:
        pnum, pcur = _parse_money(r.get("priceText"))
        tnum, tcur = _parse_money(r.get("totalText"))
        currency = pcur or tcur
        price_per_night = pnum
        price_total = tnum if tnum is not None else (round((pnum or 0.0)*nights, 2) if pnum is not None else None)
        offers.append({
            "room_name": r.get("roomName"),
            "board": r.get("boardTxt"),
            "capacity_text": r.get("capacity"),
            "price_per_night": price_per_night,
            "price_total": price_total,
            "currency": currency,
            "nights": nights,
            "raw": {
                "price_text": r.get("priceText"),
                "total_text": r.get("totalText"),
                "price_note": r.get("priceNote"),
                "room_id": r.get("roomId"),
                "rate_id": r.get("rateId"),
            }
        })
    return offers

async def crawl_once(page, loc_url: str, checkin: date, nights: int) -> Dict[str, Any]:
    print(f"[GO] {loc_url} | {checkin} +{nights}")
    await page.goto(loc_url, wait_until="domcontentloaded")
    frame = await _current_booking_frame(page)
    await _set_dates_and_search(frame, checkin, nights)

    # Ako nema nijednog izaberi odaberi brzo odustani jer nema ponuda
    any_choose = await frame.locator(
        "input[type='submit'][value*='Izaberi'], input[type='submit'][value*='Odaberi'], "
        ".button:has-text('IZABERI'), .button:has-text('ODABERI')"
    ).count()
    if not any_choose:
        print("[SKIP] no choose buttons on list")
        hotels = await _parse_hotels(frame)
        return {"hotels": hotels, "offers_by_hotel": {}}

    hotels = await _parse_hotels(frame)
    offers_by_hotel: Dict[str, List[Dict[str, Any]]] = {}

    rows = frame.locator(".property-show-box, .property-show-table, .property, .property-item, .hotel")
    nrows = await rows.count()

    for h in hotels:
        # nađi red po imenu (ako ima duplikata)
        target = None
        for i in range(nrows):
            nm = ""
            try:
                nm = (await rows.nth(i).locator(".property-name, .property-title, .hotel-name, h3, h4").first.inner_text()).strip()
            except: pass
            if nm and nm.lower() == h["name"].lower():
                target = rows.nth(i); break
        if target is None:
            target = rows.first

        # preskoči hotele bez dugmeta 
        if not await _row_has_choose_or_available(target):
            offers_by_hotel[h["name"]] = [{"info": "no rooms / no choose button on list"}]
            continue

        try:
            detail_frame = await _open_hotel_detail(page, frame, target)
            # valuta → EUR ako postoji
            try:
                if await detail_frame.locator("select[name*='currency'], #currency").count():
                    await detail_frame.select_option("select[name*='currency'], #currency", label="EUR")
                    await detail_frame.wait_for_timeout(100)
            except: pass

            offers_by_hotel[h["name"]] = await _scrape_units(detail_frame, nights)
        except Exception as e:
            offers_by_hotel[h["name"]] = [{"error": f"{type(e).__name__}: {e}"}]

        try:
            await page.goto(loc_url, wait_until="domcontentloaded")
            frame = await _current_booking_frame(page)
            await _set_dates_and_search(frame, checkin, nights)
            rows = frame.locator(".property-show-box, .property-show-table, .property, .property-item, .hotel")
            nrows = await rows.count()
        except Exception as e:
            print("[WARN] reset failed, retry hard reload:", e)
            await page.goto(loc_url, wait_until="domcontentloaded")
            frame = await _current_booking_frame(page)
            await _set_dates_and_search(frame, checkin, nights)
            rows = frame.locator(".property-show-box, .property-show-table, .property, .property-item, .hotel")
            nrows = await rows.count()

    return {"hotels": hotels, "offers_by_hotel": offers_by_hotel}

async def main(headless: bool, out: str, only_locs: List[str], dfrom: str, dto: str, nmin: int, nmax: int):
    D0, D1 = date.fromisoformat(dfrom), date.fromisoformat(dto)
    nights_list = list(range(nmin, nmax+1))

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"]
        )
        ctx = await browser.new_context(
            locale="sr-RS",
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
            viewport={"width":1280,"height":900}
        )
        ctx.set_default_timeout(9000)
        page = await ctx.new_page()
        page.set_default_timeout(9000)

        await ctx.route("**/*", lambda r: asyncio.create_task(
            r.abort() if r.request.resource_type in {"image","font","stylesheet","media"} else r.continue_()
        ))

        locs = [normalize_loc(u) for u in (only_locs or [f"{BASE}/sr/location/mariborsko-pohorje"])]

        data: Dict[str, Any] = {}
        for loc in locs:
            key = loc.rstrip("/").rsplit("/",1)[-1]
            data[key] = {"location_url": loc, "date_range": {"from": ymd(D0), "to": ymd(D1)}, "by_date": {}}

            for d in daterange_inclusive(D0, D1):
                dd = ymd(d); data[key]["by_date"][dd] = {}
                for n in nights_list:
                    print(f"\n[COMBO] {key} | {dd} | nights={n}")
                    try:
                        res = await crawl_once(page, loc, d, n)
                        data[key]["by_date"][dd][str(n)] = res
                        await page.wait_for_timeout(120)
                    except Exception as e:
                        print("[ERR] combo failed:", type(e).__name__, e)
                        data[key]["by_date"][dd][str(n)] = {"error": f"{type(e).__name__}: {e}"}

                await page.wait_for_timeout(180)

        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        await ctx.close(); await browser.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--out", default="pohorje_units.json")
    ap.add_argument("--only-locs", nargs="*", default=["https://kontiki.rs/sr/location/mariborsko-pohorje"])
    ap.add_argument("--from-date", default="2025-11-15")
    ap.add_argument("--to-date",   default="2025-11-15")
    ap.add_argument("--min-nights", type=int, default=1)
    ap.add_argument("--max-nights", type=int, default=3)
    args = ap.parse_args()
    asyncio.run(main(args.headless, args.out, args.only_locs, args.from_date, args.to_date, args.min_nights, args.max_nights))
