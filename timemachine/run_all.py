"""
run_all.py - rebuilds the whole Time Machine from the raw files in data/raw, in order:
  build_timeline -> crash_library -> narratives -> analysis -> sector_history -> montecarlo -> export
Add --validate to also rerun the 60-year walk-forward test of the Monte Carlo models (about 15-30 min).
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def main(validate=False):
    from timemachine import analysis, build_timeline, crash_library, early_warning, export, montecarlo, narratives, sector_history
    t0 = time.time()
    print("1/7 timeline ...", flush=True); build_timeline.build()
    print("2/7 crash library ...", flush=True); crash_library.build()
    print("3/7 narratives ...", flush=True); narratives.build()
    print("4/7 analysis ...", flush=True); analysis.build()
    print("5/7 sector test ...", flush=True); sector_history.build()
    print("    early-warning lights ...", flush=True); early_warning.build()
    print("6/7 Monte Carlo (today) ...", flush=True); montecarlo.forecast_today()
    if validate:
        print("    walk-forward validation ...", flush=True)
        montecarlo.score(montecarlo.walk_forward())
    print("7/7 export to Excel ...", flush=True); export.build()
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(validate="--validate" in sys.argv)
