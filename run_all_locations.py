
import argparse
import subprocess
import sys
import shlex
from pathlib import Path
from urllib.parse import urlparse

LOCATIONS = [
    # "https://kontiki.rs/sr/packages/srbija~beograd~turska~belek?SearchId=64646efc-cceb-4e3f-9fa4-160dfa595105&R1Adult=2",
    # "https://kontiki.rs/sr/packages/srbija~beograd~turska~side?SearchId=efce11f3-96f8-4424-b801-4c15f570192c&R1Adult=2",
    # "https://kontiki.rs/sr/packages/srbija~beograd~united-states~njujork?SearchId=3c626e34-ec3c-4ff4-b52b-981c61bdbe13&R1Adult=2",
    # "https://kontiki.rs/sr/packages/srbija~beograd~united-states~majami?SearchId=f06b75cb-72fd-427e-96a8-5882e0d60bc9&R1Adult=2",
    # "https://kontiki.rs/sr/packages/srbija~beograd~united-states~njujork?SearchId=eb9a0ef3-ce1c-4bc4-a548-314aaebd2894&R1Adult=2",
    # "https://kontiki.rs/sr/packages/srbija~beograd~united-states~majami?SearchId=b856ec81-1c90-4c17-96a8-bb2413384b7f&R1Adult=2",
    # "https://kontiki.rs/sr/packages/srbija~beograd~united-states~njujork?SearchId=11e38475-c449-4834-908a-1ffef735c52d&R1Adult=2",
    # "https://kontiki.rs/sr/packages/srbija~beograd~united-states~njujork?SearchId=922844ea-e363-4dfe-914e-d0e77be2192e&R1Adult=2",
    # "https://kontiki.rs/sr/location/belek",
    # "https://kontiki.rs/sr/location/bodrum",
    # "https://kontiki.rs/sr/location/cesme",
    # "https://kontiki.rs/sr/location/didim",
    # "https://kontiki.rs/sr/location/fetije",
    # "https://kontiki.rs/sr/location/istanbul",
    # "https://kontiki.rs/sr/location/izmir",
    # "https://kontiki.rs/sr/location/kas",
    # "https://kontiki.rs/sr/location/kemer",
    # "https://kontiki.rs/sr/location/kusadasi",
    # "https://kontiki.rs/sr/location/marmaris",
    # "https://kontiki.rs/sr/location/side",
    # "https://kontiki.rs/sr/location/aruba",
    # "https://kontiki.rs/sr/location/punta-kana",
    # "https://kontiki.rs/sr/location/hurgada",
    # "https://kontiki.rs/sr/location/diznilend",
    # "https://kontiki.rs/sr/location/pariz",
    # "https://kontiki.rs/sr/location/atina",
    # "https://kontiki.rs/sr/location/rim",
    # "https://kontiki.rs/sr/location/jamajka",
    # "https://kontiki.rs/sr/location/aja-napa",
    # "https://kontiki.rs/sr/location/larnaka",
    # "https://kontiki.rs/sr/location/limasol",
    # "https://kontiki.rs/sr/location/pafos",
    # "https://kontiki.rs/sr/location/protaras",
    # "https://kontiki.rs/sr/location/amsterdam",
    # "https://kontiki.rs/sr/location/muskat",
    # "https://kontiki.rs/sr/location/lisabon",
    # "https://kontiki.rs/sr/location/doha",
    # "https://kontiki.rs/sr/location/maldivi",
    # "https://kontiki.rs/sr/location/malta",
    # "https://kontiki.rs/sr/location/mauricijus",
    # "https://kontiki.rs/sr/location/sejseli",
    # "https://kontiki.rs/sr/location/barselona",
    # "https://kontiki.rs/sr/location/madrid",
    # "https://kontiki.rs/sr/location/malaga",
    # "https://kontiki.rs/sr/location/tenerife",
    # "https://kontiki.rs/sr/location/valensija",
    # "https://kontiki.rs/sr/location/koh-samui",
    # "https://kontiki.rs/sr/location/zanzibar",
    # "https://kontiki.rs/sr/location/dubai",
    # "https://kontiki.rs/sr/location/havaji",
    # "https://kontiki.rs/sr/location/bansko",
    # "https://kontiki.rs/sr/location/jahorina",
    # "https://kontiki.rs/sr/location/majami",
    # "https://kontiki.rs/sr/packages/srbija~beograd~united-states~majami?SearchId=2915bfd2-874c-444b-9be0-842ac7fe702d&R1Adult=2",
    # "https://kontiki.rs/sr/packages/srbija~beograd~united-states~njujork?SearchId=6c11a4fd-6d4a-41ea-bbfc-8662f0d13768&R1Adult=2",
    # "https://kontiki.rs/sr/packages/srbija~beograd~united-states~havaji?SearchId=751dac80-2074-4ba6-a66c-23f18462d267&R1Adult=2",
    # "https://kontiki.rs/sr/location/njujork",
]


def slug_from_location_url(u: str) -> str:

    p = urlparse(u)
    parts = [x for x in p.path.split("/") if x]
    return parts[-1] if parts else "unknown"

def build_child_cmd(
    start_url: str,
    output_file: Path,
    headless: bool,
    extra_child_args: list[str],
    pyfile: str = "scrape_kontiki_playwright.py",
) -> list[str]:

    cmd = [sys.executable, pyfile, "--start-url", start_url, "--output", str(output_file)]
    if headless:
        cmd.append("--headless")
    if extra_child_args:
        cmd.extend(extra_child_args)
    return cmd

def main():
    ap = argparse.ArgumentParser(description="Runner for scrape_kontiki_playwright.py across many locations")
    ap.add_argument("--headless", action="store_true", help="Prosledi --headless child skripti")
    ap.add_argument("--skip-existing", action="store_true", help="Preskoči lokaciju ako output fajl već postoji")
    ap.add_argument("--out-dir", default=".", help="Folder gde se upisuju json fajlovi (default: .)")
    ap.add_argument("--script", default="scrape_kontiki_playwright.py",
                    help="Ime/putanja child skripte (default: scrape_kontiki_playwright.py)")
    ap.add_argument("child_args", nargs="*", help="(Opc.) dodatni argumenti za child skriptu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    failures = 0

    for idx, loc in enumerate(LOCATIONS, 1):
        slug = slug_from_location_url(loc)
        output_path = out_dir / f"offers_kontiki_{slug}.json"

        if args.skip_existing and output_path.exists():
            print(f"[{idx}/{len(LOCATIONS)}] SKIP {slug} (postoji: {output_path.name})")
            continue

        cmd = build_child_cmd(
            start_url=loc,
            output_file=output_path,
            headless=args.headless,
            extra_child_args=args.child_args,
            pyfile=args.script,
        )

        print(f"[{idx}/{len(LOCATIONS)}] RUN {slug} → {output_path.name}")
        print(" ", shlex.join(cmd))

        try:
            proc = subprocess.run(cmd, check=False)
            if proc.returncode == 0 and output_path.exists():
                successes += 1
                print(f"  -> OK ({output_path.stat().st_size} B)")
            else:
                failures += 1
                print(f"  -> FAIL (returncode={proc.returncode})")
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Zaustavljeno od strane korisnika.")
            break
        except Exception as e:
            failures += 1
            print(f"  -> ERROR: {e}")

    print(f"\n[DONE] success={successes}, failed={failures}, total={successes+failures}")

if __name__ == "__main__":
    main()
