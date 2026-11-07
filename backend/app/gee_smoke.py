"""One-off check that YOUR Earth Engine setup works end to end (the automated tests cannot reach Google).

    python -m backend.app.gee_smoke --lat 14.28 --lon 74.44 --days 60

It initialises Earth Engine with the server settings (.env), requests Sentinel-2 scenes for a small circular field, prints what came back
(scene rows, usable observations, a map tile URL) and exits non-zero with an actionable message if anything fails.
Nothing is stored and nothing is invented: an empty result is reported as empty.
"""
import argparse
import sys
from datetime import UTC, datetime, timedelta

from .core.config import get_settings
from .services import satellite as sat
from .services.earth_engine import EarthEngineProvider
from .services.satellite import SatelliteError, SatelliteProvider


def run_smoke(provider: SatelliteProvider, lat: float, lon: float, days: int = 60, radius_m: float = 150, min_valid: float = 0.5, th: sat.Thresholds | None = None) -> dict:
    """Pure orchestration (unit-tested with a fake provider)."""
    th = th or sat.Thresholds()
    geometry = sat.validate_polygon(sat.circle_polygon(lat, lon, radius_m))["geometry"]
    end = datetime.now(UTC).date()
    rows = provider.scene_stats(geometry, end - timedelta(days=days), end)
    obs = sat.aggregate_scene_rows(rows, min_valid)
    layer = provider.map_layer(geometry, obs[-1].date, th, "classes") if obs else None
    return {"scenes_returned": len(rows), "rows": rows, "usable_observations": [o.to_dict() for o in obs], "layer": layer}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--radius", type=float, default=150, help="field radius in metres")
    a = ap.parse_args(argv)
    s = get_settings()
    print(f"Project: {s.gee_project or '(not set)'}   thresholds: low={s.ndvi_low} high={s.ndvi_high}")
    try:
        out = run_smoke(EarthEngineProvider(), a.lat, a.lon, a.days, a.radius, s.satellite_min_valid_fraction, sat.Thresholds(s.ndvi_low, s.ndvi_high))
    except SatelliteError as e:
        print(f"\nFAILED ({type(e).__name__}): {e}")
        return 1
    print(f"\nScenes returned by Earth Engine: {out['scenes_returned']}")
    for r in out["rows"][:12]:
        print(f"  {r.get('date')}  NDVI={r.get('ndvi_mean')}  clear share of field={r.get('valid_fraction')}  scene cloud%={r.get('cloud_pct')}  tile={r.get('tile')}")
    obs = out["usable_observations"]
    print(f"\nUsable observations (clear share >= {s.satellite_min_valid_fraction:.0%}): {len(obs)}")
    for o in obs:
        print(f"  {o['date']}  NDVI={o['ndvi_mean']:.3f}  clear={o['valid_fraction']:.0%}  scenes={o['scenes']}")
    if out["layer"]:
        print(f"\nMap tile URL template:\n  {out['layer']['tile_url']}")
    elif not obs:
        print("\nNo usable observation in this window. That is normal in cloudy periods; try --days 180. Nothing was estimated.")
    print("\nOK: Earth Engine is reachable and returned data." if out["scenes_returned"] else "\nOK: Earth Engine is reachable, but it returned no scenes for this window.")
    return 0


if __name__ == "__main__":
    sys.exit(main())