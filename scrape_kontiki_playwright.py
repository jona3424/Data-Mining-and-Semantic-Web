
r"""

Instalacija:
  pip install playwright selectolax
  python -m playwright install chromium

Primeri:
  # samo ispis pronađenih hotel linkova
  python scrape_kontiki_playwright.py --start-url "https://kontiki.rs/sr/location/kemer" --list-only

  # full scrape (headless), 3 meseca unapred, bez limita
  python scrape_kontiki_playwright.py --start-url "https://kontiki.rs/sr/location/kemer" --months-ahead 3 --workers 6 --headless --output offers_kontiki.json

  # sa ograničenjem ukupnog broja hotela
  python scrape_kontiki_playwright.py --start-url "https://kontiki.rs/sr/location/kemer" --limit 50 --headless
"""

import argparse
import asyncio
import json
import random
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, TimeoutError as PWTimeout
from selectolax.parser import HTMLParser

SITE = "kontiki"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
LOCALE = "sr-RS"

LIST_SCROLL_STEP = 1600
LIST_SCROLL_PAUSE_MS = 600

HOTEL_GOTO_TIMEOUT_MS = 45000
WAIT_ROOMS_TIMEOUT_MS = 20000
SETTLE_MS = 600
RANDOM_JITTER_MS = (140, 260)

def _jitter() -> int:
    return random.randint(*RANDOM_JITTER_MS)

def _clean(s: Optional[str]) -> Optional[str]:
    if not s: return None
    return re.sub(r"\s+", " ", s).strip() or None

async def click_cookies(page: Page) -> None:
    for sel in ["button:has-text('Prihv')", "button:has-text('Accept')", "text=/Prihv/i"]:
        try:
            btn = page.locator(sel).first
            if await btn.count():
                await btn.scroll_into_view_if_needed()
                await btn.click(timeout=2500)
                await page.wait_for_timeout(250)
                return
        except:
            pass

async def toggle_only_available(page: Page) -> None:
    try:
        chk = page.locator("#chkShowOnlyAvailableHotels")
        if await chk.count():
            if not await chk.is_checked():
                await chk.scroll_into_view_if_needed()
                await chk.click()
                await page.wait_for_timeout(700)
            return
    except:
        pass
    try:
        lab = page.get_by_text(re.compile(r"Prikaži.*raspoloživ", re.I)).first
        if await lab.count():
            await lab.scroll_into_view_if_needed()
            await lab.click()
            await page.wait_for_timeout(700)
    except:
        pass

async def _harvest_page_links(page: Page, base_url: str, bag: Set[str], per_page_limit: Optional[int]=None):
    anchors = page.locator("a[href*='hotel/']")
    n = await anchors.count()
    for i in range(n):
        href = await anchors.nth(i).get_attribute("href")
        if not href or href.startswith("javascript:"): continue
        full = urljoin(base_url, href)
        if "hotel/" in full and full not in bag:
            bag.add(full)
            if per_page_limit and len(bag) >= per_page_limit:
                return

async def _go_through_pagination(page: Page, base_url: str, bag: Set[str], per_page_limit: Optional[int]=None):
    """Next kroz stranice i na svakoj strani dohvata hotel linkove."""
    while True:
        await page.wait_for_timeout(300 + _jitter())
        await _harvest_page_links(page, base_url, bag, per_page_limit)
        next_li = page.locator("#pagination li.page-item.next:not(.disabled)")
        if not await next_li.count():
            break
        try:
            btn = next_li.locator("a.page-link")
            await btn.click()
            # sačekaj da se lista zameni
            await page.wait_for_timeout(900 + _jitter())
            await page.mouse.wheel(0, LIST_SCROLL_STEP)
            await page.wait_for_timeout(LIST_SCROLL_PAUSE_MS + _jitter())
        except:
            break

async def _iterate_dates_after_current(page: Page, start_url: str, months_ahead: int, bag: Set[str], limit: Optional[int]):
    """ide kroz sve ponudjene datume nakon trenutnog, do cutoff-a"""
    today = datetime.now().date()
    cutoff = today + timedelta(days=30*months_ahead)
    seen_dates = set()

    next_arrow_sel = ".swiper-button-next, .swiper-container .swiper-button-next"

    while True:
        items = page.locator(".swiper-slide a:has(.checkindate)")
        cnt = await items.count()
        progressed = False

        for i in range(cnt):
            a = items.nth(i)
            try:
                label = await a.inner_text()
            except:
                continue
            m = re.search(r"(\d{2}\.\d{2}\.\d{4})", label)
            if not m:
                continue
            d = datetime.strptime(m.group(1), "%d.%m.%Y").date()
            if d in seen_dates:
                continue
            if d <= today:  # preskoči trenutni i prošle jer smo njih već pokrili
                seen_dates.add(d)
                continue
            if d > cutoff:
                continue

            try:
                await a.scroll_into_view_if_needed()
                await a.click()
                seen_dates.add(d)
                progressed = True

                # čekaj refresh i osiguraj “samo raspoložive”
                await page.wait_for_timeout(1000 + _jitter())
                await toggle_only_available(page)

                await _go_through_pagination(page, start_url, bag, None)
                if limit and len(bag) >= limit:
                    return
            except:
                continue

        if limit and len(bag) >= limit:
            return

        if not progressed:
            arrow = page.locator(next_arrow_sel)
            if await arrow.count():
                try:
                    await arrow.click()
                    await page.wait_for_timeout(600 + _jitter())
                    continue
                except:
                    pass
            break

async def discover_listing_links_full(page: Page, start_url: str, months_ahead: int, limit: Optional[int]) -> List[str]:
    await page.goto(start_url, wait_until="domcontentloaded")
    await click_cookies(page)
    await toggle_only_available(page)
    await page.wait_for_timeout(500 + _jitter())

    bag: Set[str] = set()

    # 1) sve stranice za trenutni datum
    await _go_through_pagination(page, start_url, bag, None)
    if limit and len(bag) >= limit:
        return list(bag)[:limit]

    # 2) ostali datumi 
    await _iterate_dates_after_current(page, start_url, months_ahead, bag, limit)

    return list(bag)[:limit] if limit else list(bag)

# meta sa hotel HTML stranice
def parse_meta_from_html(html: str) -> Dict[str, Optional[str]]:
    doc = HTMLParser(html)
    title = None
    for sel in ("h1", ".hotel-name", ".offer-title"):
        el = doc.css_first(sel)
        if el and _clean(el.text()):
            title = _clean(el.text()); break

    location = None
    for sel in (".hotel-town", ".hotel-location", ".hotel-city"):
        el = doc.css_first(sel)
        if el and _clean(el.text()):
            location = _clean(el.text()); break

    stars_raw = None
    stars = doc.css(".hotel-category .fa-star, .hotel-category i.fa-star")
    if stars: stars_raw = str(len(stars))

    departure_date_raw = nights = None
    for sel in (
        ".book-detail .date-x span.date[data-type='Night']",
        ".book-detail .date-x .date[data-type='Night']",
        "span.date[data-type='Night'][data-count]"
    ):
        el = doc.css_first(sel)
        if not el: continue
        txt = _clean(el.text()) or _clean(el.attributes.get("textContent", ""))
        if txt:
            m = re.search(r"(\d{2}\.\d{2}\.\d{4})", txt)
            if m: departure_date_raw = m.group(1)
            nattr = (el.attributes.get("data-count") or "").strip()
            if nattr.isdigit() and int(nattr) > 0:
                nights = nattr
            else:
                m2 = re.search(r"(\d+)\s*No", txt, flags=re.I)
                if m2: nights = m2.group(1)
        if departure_date_raw or nights: break

    return {"title": title, "location": location, "stars_raw": stars_raw,
            "departure_date_raw": departure_date_raw, "nights": nights}

# DOM parsing soba: EUR + RSD 
async def extract_rooms_from_dom(page: Page) -> List[Dict[str, Any]]:
    try:
        await page.wait_for_selector(
            ".room-item .col-price .price-number, .room-item .col-price .price:not(.oldprice)",
            timeout=WAIT_ROOMS_TIMEOUT_MS
        )
    except PWTimeout:
        return []

    for _ in range(6):
        await page.mouse.wheel(0, 1200)
        await page.wait_for_timeout(220 + _jitter())

    rooms = await page.evaluate("""
    () => {
      function clean(t){return t ? t.replace(/\\s+/g,' ').trim() : null;}
      function numFromParts(numEl, penEl){
        if(!numEl && !penEl) return null;
        const n = numEl ? clean(numEl.textContent) : null;
        const p = penEl ? clean(penEl.textContent) : null;
        if(!n) return null;
        const whole = parseInt(n.replace(/[\\.\\s]/g,''), 10);
        const cents = p ? parseInt(p.replace(/\\D/g,''),10) : 0;
        if (isNaN(whole)) return null;
        return whole + (isNaN(cents)?0:cents)/100.0;
      }
      function curCode(el){
        if(!el) return null;
        const c = (el.getAttribute('data-currency-code') || '').trim().toLowerCase();
        return c || null;
      }
      function parseExchange(el){
        if(!el) return {amount:null, currency:null};
        const txt = clean(el.textContent) || '';
        const mr = txt.match(/([\\d\\.,\\s]+)\\s*(RSD|rsd|дин)/);
        const me = txt.match(/([\\d\\.,\\s]+)\\s*(EUR|eur|€)/);
        if(mr){
          let a = mr[1].replace(/\\.(?=\\d{3}(?:\\D|$))/g,'').replace(',', '.').replace(/\\s/g,'');
          const v = parseFloat(a);
          return {amount: isNaN(v)? null : v, currency: 'rsd'};
        }
        if(me){
          let a = me[1].replace(/\\.(?=\\d{3}(?:\\D|$))/g,'').replace(',', '.').replace(/\\s/g,'');
          const v = parseFloat(a);
          return {amount: isNaN(v)? null : v, currency: 'eur'};
        }
        return {amount:null, currency:null};
      }

      const out = [];
      document.querySelectorAll('.room-item .row').forEach(row=>{
        const rt = row.querySelector('.room-type');
        let room_type = null;
        if(rt){
          const e = rt.querySelector('.roomname .ng-binding, .roomname span, .roomname');
          room_type = clean(e ? e.textContent : rt.textContent);
          if(room_type) room_type = room_type.replace(/^\\s*Tip sobe\\s*/i, '').trim();
        }

        const b = row.querySelector('.board');
        let board = null;
        if(b){
          const be = b.querySelector('span.board');
          board = clean(be ? be.textContent : b.textContent);
          if(board) board = board.replace(/^\\s*Usluga\\s*/i, '').trim();
        }

        const col = row.querySelector('.col-price');
        if(!col) return;

        // old price
        let old_price_eur = null, old_price_rsd = null;
        const oldEl = col.querySelector('span.price.oldprice');
        if(oldEl){
          const on = oldEl.querySelector('.price-number');
          const op = oldEl.querySelector('.price-penny');
          const oc = oldEl.querySelector('.price-currency');
          const oval = numFromParts(on, op);
          const code = curCode(oc);
          if(oval !== null){
            if(code === 'eur') old_price_eur = oval;
            else if(code === 'rsd') old_price_rsd = oval;
            else old_price_eur = oval; // default EUR
          }
        }

        // current price
        let price_eur = null, price_rsd = null;
        const cur = col.querySelector('span.price.ng-binding:not(.oldprice), span.price:not(.oldprice)');
        if(cur){
          const cn = cur.querySelector('.price-number');
          const cp = cur.querySelector('.price-penny');
          const cc = cur.querySelector('.price-currency');
          const cval = numFromParts(cn, cp);
          const code = curCode(cc);
          if(cval !== null){
            if(code === 'eur') price_eur = cval;
            else if(code === 'rsd') price_rsd = cval;
            else price_eur = cval;
          }
        }

        // exchange price (obično RSD)
        const exch = col.querySelector('.exchange-price');
        if(exch){
          const ex = parseExchange(exch);
          if(ex.currency === 'rsd' && price_rsd === null) price_rsd = ex.amount;
          if(ex.currency === 'eur' && price_eur === null) price_eur = ex.amount;
        }

        if(room_type || board || price_eur !== null || price_rsd !== null){
          out.push({
            room_type, board,
            old_price_eur, old_price_rsd,
            price_eur, price_rsd
          });
        }
      });
      return out;
    }
    """)
    return rooms

async def parse_hotel(page: Page, url: str) -> Dict[str, Any]:
    await page.goto(url, wait_until="domcontentloaded", timeout=HOTEL_GOTO_TIMEOUT_MS)
    await click_cookies(page)
    await page.wait_for_timeout(SETTLE_MS + _jitter())

    for _ in range(6):
        await page.mouse.wheel(0, 1200)
        await page.wait_for_timeout(220 + _jitter())

    html = await page.content()
    meta = parse_meta_from_html(html)
    rooms = await extract_rooms_from_dom(page)

    return {
        "site": SITE,
        "url": url,
        "title": meta.get("title"),
        "location": meta.get("location"),
        "stars_raw": meta.get("stars_raw"),
        "departure_date_raw": meta.get("departure_date_raw"),
        "nights": meta.get("nights"),
        "rooms": rooms,
    }

async def run_scrape(start_url: str, months_ahead: int, limit: int, output: str, headless: bool, list_only: bool, workers: int):
    async with async_playwright() as pw:
        browser: Browser = await pw.chromium.launch(headless=headless)
        context: BrowserContext = await browser.new_context(user_agent=UA, locale=LOCALE)
        page: Page = await context.new_page()

        links = await discover_listing_links_full(page, start_url, months_ahead, (limit or None))
        print(f"[MAIN] Discovered {len(links)} hotel links" + (f" (limited to {limit})" if limit else ""))

        if list_only:
            for u in links: print(u)
            await context.close(); await browser.close(); return

        if limit:
            links = links[:limit]

        sem = asyncio.Semaphore(max(1, workers))
        results: List[Dict[str, Any]] = []

        async def worker(u: str):
            async with sem:
                p = await context.new_page()
                try:
                    item = await parse_hotel(p, u)
                    results.append(item)
                    print(f"[OK] {item.get('title') or u} -> {len(item.get('rooms') or [])} rooms")
                finally:
                    await p.close()

        await asyncio.gather(*(worker(u) for u in links))

        with open(output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"[DONE] Wrote {len(results)} hotels to {output}")

        await context.close(); await browser.close()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start-url", default="https://kontiki.rs/sr/location/kemer")
    p.add_argument("--months-ahead", type=int, default=3, help="Koliko meseci unapred iterirati datume")
    p.add_argument("--limit", type=int, default=0, help="Maks. ukupno hotel linkova (0 = svi)")
    p.add_argument("--output", default="offers_kontiki.json")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--list-only", action="store_true")
    p.add_argument("--workers", type=int, default=6)
    return p.parse_args()

def main():
    args = parse_args()
    asyncio.run(run_scrape(
        args.start_url,
        args.months_ahead,
        args.limit if args.limit > 0 else 0,
        args.output,
        args.headless,
        args.list_only,
        args.workers
    ))

if __name__ == "__main__":
    main()
