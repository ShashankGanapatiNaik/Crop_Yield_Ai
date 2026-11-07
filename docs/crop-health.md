# Satellite crop-health monitoring (Sentinel-2 NDVI)

Farmers mark a field on a map and see a colour-coded NDVI map, the latest observation, a plain-language explanation, and the NDVI history of that field.
NDVI is a **greenness indicator, not a diagnosis**: the app never claims to identify a disease, pest or nutrient problem.

> **Verification status.** Everything except the live call to Google is covered by tests (237 backend tests, plus a browser run of the page with the API
> mocked). **Earth Engine itself could not be reached from the development sandbox**, so the Earth Engine expressions have **not** been run against
> Google's servers. Run the one-time smoke test in step 5 on your machine before relying on the feature.

## How it works

```
React page ─► /crop-health/fields  (validated geometry, owner-only)        services/satellite.py     pure logic, provider-independent
            ─► /crop-health/fields/{id}/latest  ─┐                          services/earth_engine.py  the ONLY module that imports `ee`
            ─► /crop-health/fields/{id}/history ─┴► services/crop_health.py  DB cache + orchestration (no Earth Engine code)
```

* **Dataset:** `COPERNICUS/S2_SR_HARMONIZED` (Sentinel-2 Level-2A surface reflectance, 10 m). The *harmonized* collection shifts scenes processed after
  25 Jan 2022 into the range of older ones, so NDVI stays comparable across years.
* **NDVI:** `(B8 - B4) / (B8 + B4)` = (NIR - Red) / (NIR + Red). The 0.0001 reflectance scale cancels out.
* **Cloud and shadow masking:** per pixel, using the scene classification band `SCL`: saturated/defective (1), cloud shadow (3), medium and high cloud
  probability (8, 9), thin cirrus (10) and snow/ice (11) are removed. Whole scenes above `SATELLITE_SCENE_MAX_CLOUD_PCT` (80 %) are skipped up front.
* **Per-scene statistics:** mean NDVI of the cloud-free pixels inside the field, plus the share of the field those pixels cover
  (`valid_fraction`, approximate to about one pixel at the boundary). Scenes covering less than `SATELLITE_MIN_VALID_FRACTION` (50 %) are discarded.
* **Aggregation:** scenes of the same date (neighbouring tiles) are averaged, weighted by clear share. One observation per date is stored.
* **Map layer:** the NDVI of the latest observation date, clipped to the field, rendered as three classes that match the legend
  (red / yellow / green) or as a gradient (`?style=gradient`). Tile URLs are issued by Earth Engine and are **temporary**.
* **Caching:** results are stored in `ndvi_observations` (default 12 h), so opening the page repeatedly does not hit Earth Engine. `?refresh=true` forces a new
  request. If a refresh fails and cached data exist, the cached data are returned flagged `stale: true` with a warning.
* **No imagery is never faked:** if no scene is clear enough, the API returns `status: "no_data"` with a reason (common in the monsoon); no value, trend or map is invented.

## NDVI colours and interpretation

| Class | NDVI (default) | Colour | Typical meaning (not a diagnosis) |
|---|---|---|---|
| Low greenness | below 0.2 | red | bare soil, newly sown or harvested/fallow field, sparse crop, or stress |
| Moderate / declining | 0.2 to 0.5 | yellow | early or late growth stage, partial canopy, possible stress |
| High greenness | 0.5 and above | green | dense, green vegetation |

The boundaries are **approximate and configurable** (`NDVI_LOW`, `NDVI_HIGH`). They differ by crop and growth stage, so compare a field with itself over
time, or with the same crop at the same stage. A fall of 0.10 or more between observations is described as "can come from ripening, harvest or stress",
never as a specific cause.

## API

All endpoints except `/crop-health/status` need `Authorization: Bearer <token>`; fields are private to their owner.

| Method and path | Purpose |
|---|---|
| `GET /crop-health/status` | Is satellite monitoring configured? Thresholds, limits, legend. No secrets. |
| `POST /crop-health/fields` | Save a field. Body: `{"name", "geometry": GeoJSON Polygon}` **or** `{"name", "latitude", "longitude", "radius_m"}` |
| `GET /crop-health/fields`, `GET /crop-health/fields/{id}` | List / read your fields |
| `DELETE /crop-health/fields/{id}` | Delete a field and its cached observations |
| `GET /crop-health/fields/{id}/latest?days=90&refresh=false&style=classes` | Latest observation, previous one, interpretation, legend, map layer |
| `GET /crop-health/fields/{id}/history?days=180` | Chronological NDVI observations with their class |

Geometry rules (HTTP 422 with `{"detail": "Validation failed", "errors": [{"field", "message"}]}` when broken): GeoJSON order is **[longitude, latitude]**;
latitude -56 to 84 (Sentinel-2 coverage); one outer ring (no holes); 3 to 500 distinct points; no self-intersections; area between `FIELD_MIN_HA` (0.05) and
`FIELD_MAX_HA` (1000); extent under about 25 km. Satellite failures return 502 (service error), 503 (not configured) or 504 (timeout).

```bash
curl -X POST localhost:8000/crop-health/fields -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"name":"North paddy","latitude":14.28,"longitude":74.44,"radius_m":150}'
curl "localhost:8000/crop-health/fields/1/latest?days=90"  -H "Authorization: Bearer $TOKEN"
```

Shape of a `latest` response (field names are real; the values below are placeholders, **not** satellite results):

```json
{ "status": "ok", "observation": {"date": "YYYY-MM-DD", "ndvi_mean": 0.0, "valid_fraction": 0.0, "category": "low|moderate|high", "color": "#hex"},
  "previous": {"...": "same shape or null"}, "interpretation": {"summary": "...", "caveats": ["..."], "disclaimer": "..."},
  "layer": {"tile_url": "https://.../{z}/{x}/{y}", "date": "YYYY-MM-DD", "bounds": [[south, west], [north, east]]},
  "thresholds": {"low": 0.2, "high": 0.5}, "legend": ["..."], "from_cache": false, "stale": false, "window": {"start": "...", "end": "...", "days": 90} }
```
`status: "no_data"` replaces `observation`, `previous`, `interpretation` and `layer` with `null` and adds a `reason`.

## Setup (Google Earth Engine)

The server needs a **Google Cloud project registered for Earth Engine**; the Python client always requires one. Noncommercial use is free within Google's
quota tiers; commercial use needs a commercial registration. See Google's pages on
[Cloud project setup](https://developers.google.com/earth-engine/cloud/earthengine_cloud_project_setup), [noncommercial tiers](https://developers.google.com/earth-engine/guides/noncommercial_tiers)
and [usage limits](https://developers.google.com/earth-engine/guides/usage).

**1. Project and API.** Create (or choose) a Google Cloud project, **register it for Earth Engine**, and **enable the Earth Engine API** on it.
Put the project ID in `.env`: `GEE_PROJECT=your-project-id`.

**2. Choose how the server authenticates** (pick one; the app tries them in this order):

| Option | When | Steps |
|---|---|---|
| **A. Your own login (local development)** | your laptop | `pip install -r requirements.txt` then `earthengine authenticate` once. Leave the key settings empty. |
| **B. Service account key** | a server not on Google Cloud | Cloud Console > IAM & Admin > Service accounts > create one; grant it the **Earth Engine Resource Viewer** role (and **Service Usage Consumer** if Google reports a permission error) on the project; Keys > Add key > JSON. Set `GEE_SERVICE_ACCOUNT_JSON` to the key contents on one line (or `GEE_SERVICE_ACCOUNT_KEY_FILE` to its path, optionally `GEE_SERVICE_ACCOUNT_EMAIL`). |
| **C. Application Default Credentials** | Cloud Run / Compute Engine | Attach a service account with the roles above to the service. Leave all key settings empty: no key file is needed. |

**3. Keep secrets server-side.** The key lives only in server environment variables or a secret manager. It is never sent to the browser, never logged
(settings print it as hidden), and must **never be committed**: `.env` is git-ignored; check with `git status` before committing.

**4. Install and start** (Python 3.11+):

```bash
pip install -r requirements.txt                      # includes earthengine-api
uvicorn backend.app.main:app --reload                # new tables are created automatically
cd frontend && npm install && npm run dev            # page: http://localhost:5173/crop-health
```

**5. Smoke-test once (recommended).** Proves your project, credentials and the dataset query work, using a small circle at your coordinates:

```bash
python -m backend.app.gee_smoke --lat 14.28 --lon 74.44 --days 60
```
It prints the scenes Earth Engine returned, the usable observations, and a tile URL. "No usable observation" in a cloudy period is a valid result; try `--days 180`.

## Troubleshooting

| What you see | Meaning and fix |
|---|---|
| Page banner "not set up on this server" / HTTP 503 | `GEE_PROJECT` is empty, or initialisation failed. Check the project is registered, the Earth Engine API is enabled, and the credentials have access. Run the smoke test. |
| 503 mentioning permission | Grant the service account the Earth Engine Resource Viewer (and Service Usage Consumer) role on the project. |
| 502 "usage limit reached" | Earth Engine quota. Wait, shorten the window, or rely on the 12 h cache. |
| 504 timeout | Field too large or window too long. Try a smaller field or fewer days. |
| "No clear satellite image found" | Not an error: clouds covered the field in that window (monsoon). Widen the window or retry later. |
| Map tiles stop loading | Earth Engine tile links are temporary: press **Refresh from satellite**. |
| Everything works but the colours look wrong for my crop | The class boundaries are generic. Adjust `NDVI_LOW` / `NDVI_HIGH`, and compare a field with its own history. |

## Limitations (please keep these in any report)

1. **Clouds.** In the Indian monsoon, clear scenes can be weeks apart or absent; the page then says so. The SCL mask is good but imperfect (thin haze and small
   clouds can slip through; bright surfaces can be masked wrongly). A cloud-probability product (e.g. Cloud Score+) could be added for stricter masking.
2. **Resolution.** 10 m pixels: small, narrow or irregular fields and edges mix with their surroundings.
3. **Not a diagnosis.** NDVI cannot separate stress from growth stage, bare soil or harvest, and cannot identify diseases or pests.
4. **Approximate colour thresholds** and an **approximate clear-sky share** (pixel counts vs. polygon area).
5. **Latency.** Sentinel-2 revisits about every 5 days; data appear after processing. Each fresh request takes seconds and uses Earth Engine quota.
6. **Not yet verified live** (see the note at the top): run the smoke test and report any error text.
7. **Not linked to the yield model.** NDVI is shown separately; feeding it into predictions is future work.

## Data licence and attribution

Sentinel-2 data are produced by ESA/Copernicus. The map shows the attribution **"Contains modified Copernicus Sentinel data"**; keep it visible. Map tiles
are © OpenStreetMap contributors.

## Tests

* `tests/test_satellite_core.py`: NDVI formula (including scale invariance and undefined pixels), thresholds, colour classes, coordinate and polygon
  validation (swapped/out-of-range values, holes, self-intersection, size limits), aggregation, no-data handling, cautious wording (no diagnosis terms).
* `tests/test_satellite_provider.py`: Earth Engine provider against a **fake `ee`** module: credential selection (JSON / file / user login), secret hygiene,
  band choice (B8, B4), cloud-mask classes, error translation (timeout, quota, permission, network), map layer thresholds and colours.
* `tests/test_satellite_api.py`: endpoints with a scripted fake provider: validation, ownership, caching, stale fallback, no-data, failures (502/503/504), history, deletion.