"""Google Earth Engine provider for Sentinel-2 NDVI. The ONLY module that imports `ee`.

Dataset  : COPERNICUS/S2_SR_HARMONIZED (Level-2A surface reflectance, 10 m bands B4 = Red, B8 = NIR).
           The HARMONIZED collection shifts post-2022-01-25 scenes (processing baseline >= 04.00) into the range of older scenes,
           so NDVI stays comparable across years.
Cloud    : per-pixel mask from the Scene Classification Layer (SCL): 1 saturated/defective, 3 cloud shadow, 8/9 cloud (medium/high
           probability), 10 thin cirrus, 11 snow/ice are removed. Scenes with too little clear sky over the field are dropped later
           (satellite.aggregate_scene_rows).
NDVI     : (B8 - B4) / (B8 + B4) = normalizedDifference(['B8', 'B4']); the 0.0001 reflectance scale cancels out.

Credentials are read from server settings only (never sent to the browser). Order of preference:
  1. GEE_SERVICE_ACCOUNT_JSON  (JSON key contents)        2. GEE_SERVICE_ACCOUNT_KEY_FILE (+ optional GEE_SERVICE_ACCOUNT_EMAIL)
  3. user credentials stored by `earthengine authenticate` (local development)
A Google Cloud project (GEE_PROJECT) registered for Earth Engine is always required.
"""
import json
import logging
import threading
from datetime import date, timedelta

from ..core.config import Settings, get_settings
from .satellite import COLORS, GRADIENT_PALETTE, LAYER_STYLES, SatelliteError, SatelliteNotConfigured, SatelliteTimeout, Thresholds

log = logging.getLogger("earth_engine")

COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
NIR_BAND, RED_BAND = "B8", "B4"
SCL_EXCLUDE = (1, 3, 8, 9, 10, 11)
SCALE_M = 10
TILE_EXPIRY_NOTE = "Map tile links are temporary (Earth Engine issues short-lived URLs). Refresh the analysis if tiles stop loading."


def _import_ee():
    """Separate function so tests can substitute a fake `ee` module and so the app starts without Earth Engine installed."""
    try:
        import ee
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise SatelliteNotConfigured("The 'earthengine-api' package is not installed on the server (pip install earthengine-api).") from e
    return ee


class EarthEngineProvider:
    def __init__(self, settings: Settings | None = None, ee_module=None):
        self._settings = settings
        self._ee = ee_module
        self._ready = False
        self._lock = threading.Lock()

    @property
    def settings(self) -> Settings:
        return self._settings or get_settings()

    # ---- authentication / initialisation ---------------------------------------------------------------------------
    def _build_credentials(self, ee):
        """Returns credentials or None (= use the credentials saved by `earthengine authenticate`)."""
        s = self.settings
        key_json = s.gee_service_account_json.get_secret_value().strip()
        if key_json:
            try:
                json.loads(key_json)
            except ValueError as e:
                raise SatelliteNotConfigured("GEE_SERVICE_ACCOUNT_JSON is not valid JSON.") from e
            return ee.ServiceAccountCredentials(s.gee_service_account_email or None, key_data=key_json)
        if s.gee_service_account_key_file.strip():
            from pathlib import Path
            if not Path(s.gee_service_account_key_file).is_file():
                raise SatelliteNotConfigured("GEE_SERVICE_ACCOUNT_KEY_FILE does not point to an existing file.")
            return ee.ServiceAccountCredentials(s.gee_service_account_email or None, key_file=s.gee_service_account_key_file)
        return None

    def ensure_ready(self):
        """Initialise once (thread-safe). Raises SatelliteNotConfigured / SatelliteError with an actionable message; never leaks secrets."""
        if self._ready:
            return self._ee
        with self._lock:
            if self._ready:
                return self._ee
            s = self.settings
            if not s.gee_configured:
                raise SatelliteNotConfigured("Satellite monitoring is not configured: set GEE_PROJECT (and credentials) on the server. See docs/crop-health.md.")
            ee = self._ee or _import_ee()
            try:
                creds = self._build_credentials(ee)
                if creds is not None:
                    ee.Initialize(credentials=creds, project=s.gee_project)
                else:
                    ee.Initialize(project=s.gee_project)
            except SatelliteError:
                raise
            except Exception as e:  # noqa: BLE001 - ee raises EEException, google.auth errors, OSError for missing credentials...
                log.warning("Earth Engine initialisation failed: %s", type(e).__name__)
                raise SatelliteNotConfigured(
                    "Could not initialise Earth Engine. Check GEE_PROJECT, that the project is registered for Earth Engine, and that the credentials "
                    "(service account or `earthengine authenticate`) have access. See docs/crop-health.md.") from e
            self._ee, self._ready = ee, True
            log.info("Earth Engine initialised for project %s", s.gee_project)
            return ee

    # ---- Earth Engine building blocks (server-side expressions) ----------------------------------------------------
    @staticmethod
    def _mask_clouds(img):
        scl = img.select("SCL")
        bad = scl.eq(SCL_EXCLUDE[0])
        for c in SCL_EXCLUDE[1:]:
            bad = bad.Or(scl.eq(c))
        return img.updateMask(bad.Not())

    @staticmethod
    def _ndvi(img):
        return img.normalizedDifference([NIR_BAND, RED_BAND]).rename("NDVI")

    def _wrap(self, ee, fn):
        """Run an Earth Engine call and translate failures into SatelliteError subclasses (no stack traces or secrets to the client)."""
        try:
            return fn()
        except SatelliteError:
            raise
        except ee.EEException as e:
            msg = str(e)
            low = msg.lower()
            log.warning("Earth Engine error: %s", msg[:300])
            if "timed out" in low or "deadline" in low:
                raise SatelliteTimeout("Earth Engine took too long. Try a smaller field or a shorter date range.") from e
            if "quota" in low or "too many requests" in low or ("rate" in low and "limit" in low):
                raise SatelliteError("Earth Engine usage limit reached. Please try again later.") from e
            if "permission" in low or "not registered" in low or "not authorized" in low or "403" in low:
                raise SatelliteNotConfigured("Earth Engine rejected the request (permission). Check the project registration and roles. See docs/crop-health.md.") from e
            raise SatelliteError("Earth Engine could not process the request.") from e
        except (TimeoutError, ConnectionError, OSError) as e:
            log.warning("Earth Engine network error: %s", type(e).__name__)
            raise SatelliteError("Could not reach Earth Engine. Check the server's internet connection.") from e

    # ---- public interface (see satellite.SatelliteProvider) --------------------------------------------------------
    def scene_stats(self, geometry: dict, start: date, end: date) -> list[dict]:
        """One row per Sentinel-2 scene intersecting the field: {date, ndvi_mean, valid_fraction, cloud_pct, tile}.
        valid_fraction = cloud-free NDVI pixels x 100 m2 / field area (capped at 1): approximate to about one pixel at the boundary."""
        ee = self.ensure_ready()
        s = self.settings

        def run():
            geom = ee.Geometry(geometry)
            field_area_m2 = geom.area(1)
            col = (ee.ImageCollection(COLLECTION).filterBounds(geom).filterDate(start.isoformat(), (end + timedelta(days=1)).isoformat())
                   .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", s.satellite_scene_max_cloud_pct)))

            def per_scene(img):
                ndvi = self._ndvi(self._mask_clouds(img))
                mean = ndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=SCALE_M, maxPixels=1e9).get("NDVI")
                count = ndvi.reduceRegion(reducer=ee.Reducer.count(), geometry=geom, scale=SCALE_M, maxPixels=1e9).get("NDVI")
                valid_area = ee.Number(ee.Algorithms.If(count, count, 0)).multiply(SCALE_M * SCALE_M)
                return ee.Feature(None, {"date": img.date().format("YYYY-MM-dd"), "ndvi_mean": mean,
                                         "valid_fraction": valid_area.divide(field_area_m2).min(1),
                                         "cloud_pct": img.get("CLOUDY_PIXEL_PERCENTAGE"), "tile": img.get("MGRS_TILE")})

            return ee.FeatureCollection(col.map(per_scene)).getInfo()

        info = self._wrap(ee, run)
        rows = [f.get("properties", {}) for f in (info or {}).get("features", [])]
        log.info("Earth Engine returned %d scenes for %s..%s", len(rows), start, end)
        return rows

    def map_layer(self, geometry: dict, obs_date: date, th: Thresholds, style: str = "classes") -> dict:
        """Map tiles of the NDVI of one date (all same-day tiles mosaicked), clipped to the field."""
        if style not in LAYER_STYLES:
            raise ValueError(f"style must be one of {LAYER_STYLES}")
        ee = self.ensure_ready()

        def run():
            geom = ee.Geometry(geometry)
            day = ee.Date(obs_date.isoformat())
            col = ee.ImageCollection(COLLECTION).filterBounds(geom).filterDate(day, day.advance(1, "day"))
            ndvi = col.map(lambda i: self._ndvi(self._mask_clouds(i))).mosaic().clip(geom)
            if style == "gradient":
                image, vis = ndvi, {"min": 0.0, "max": 0.9, "palette": GRADIENT_PALETTE}
            else:  # three classes that match the legend and the configurable thresholds exactly
                image = ee.Image(0).where(ndvi.gte(th.low), 1).where(ndvi.gte(th.high), 2).updateMask(ndvi.mask())
                vis = {"min": 0, "max": 2, "palette": [COLORS["low"], COLORS["moderate"], COLORS["high"]]}
            return image.getMapId(vis)

        mp = self._wrap(ee, run)
        url = mp["tile_fetcher"].url_format
        return {"tile_url": url, "style": style, "date": obs_date.isoformat(), "attribution": "Contains modified Copernicus Sentinel data",
                "expires_note": TILE_EXPIRY_NOTE}


_provider: EarthEngineProvider | None = None


def get_provider() -> EarthEngineProvider:
    global _provider
    if _provider is None:
        _provider = EarthEngineProvider()
    return _provider