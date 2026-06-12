# Data Directory

Datasets are not redistributed with this artifact; both are public.

## raw/li_cao_imc2020/
**Source:** Li & Cao, "Who Touched My Browser Fingerprint?", IMC 2020
**Zenodo DOI:** https://zenodo.org/records/7743719
**File:** final_with_header.csv.zip (3.7 GB compressed)

Run `bash scripts/download_data.sh` from the repository root to download and
extract. Separator: tab (`\t`).

### Schema (from Table 1 of the IMC 2020 paper)

| Feature Group | Feature Names |
|---|---|
| HTTP Headers | user_agent, browser, os, device, accept, encoding, language, timezone, http_header_list |
| Browser Features | plugins, cookie_support, webgl_support, local_storage_support, add_behavior_support, open_database_support |
| OS Features | language_list, font_list, canvas_images |
| Hardware Features | gpu_vendor, gpu_renderer, gpu_type, cpu_cores, audio_card_info, screen_resolution, color_depth, cpu_class, pixel_ratio |
| IP Features | ip_city, ip_region, ip_country |
| Consistency Features | lang_consistency, resolution_consistency, os_consistency, browser_consistency, gpu_images |

### Key statistics
- 960,853 dynamics records; 1,329,927 distinct browser instances
- Collected Jul 2017 - Jul 2018 on a European website

## raw/fpstalker/
**Source:** Vastel et al., "FP-STALKER: Tracking Browser Fingerprint
Evolutions", IEEE S&P 2018.
Obtain the public fingerprint dataset from the authors' release
(https://github.com/Spirals-Team/FPStalker), place it here as
`fingerprints.csv`, then run `python scripts/preprocess_fpstalker.py`.
