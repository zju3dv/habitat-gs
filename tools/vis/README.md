# Visualization Tools

## `rerun_nav_viewer.py`

Live `rerun` viewer for habitat-gs navigation sessions.

Default ports are read from:

[`tools/vis/config.yaml`](./config.yaml)

CLI flags override values from the config file.

Important: Always point the viewer to the same bridge as your active nav loop. In proxy-enabled shells, ensure localhost is excluded (`NO_PROXY=127.0.0.1,localhost`).

## Default layout

The viewer configures a two-page Rerun blueprint automatically:

- **Page 1 — Visuals**: 2×2 grid (RGB, Depth, Third Person RGB, BEV)
- **Page 2 — Text**: status panels (Nav Status, Metrics, Lifecycle, Errors, BEV Status, Depth Analysis)

Headless server usage:

```bash
python3 tools/vis/rerun_nav_viewer.py
```

If `--loop-id` is omitted, the viewer auto-selects the most recent active nav loop.

This defaults to writing a recording file to:

```bash
/tmp/habitat-gs-nav.rrd
```

Explicit output path:

```bash
python3 tools/vis/rerun_nav_viewer.py --loop-id <loop_id> --save /tmp/my_nav.rrd
```

If you have a remote rerun viewer reachable over TCP:

```bash
python3 tools/vis/rerun_nav_viewer.py --loop-id <loop_id> --connect <host>:<port>
```

If your installed `rerun-sdk` supports the web viewer, you can serve it directly from the server:

```bash
python3 tools/vis/rerun_nav_viewer.py --bridge-port 18911 --serve-web 9091 --grpc-port 19091
```

Then forward both the web port and the Rerun gRPC/proxy port to your local machine:

```bash
ssh -L 9091:127.0.0.1:9091 -L 19091:127.0.0.1:19091 <user>@<server>
```

Open the web UI in your browser:

```text
http://127.0.0.1:9091
```

The web viewer does not auto-connect when launched headlessly. In the viewer:

1. Click `+` in the `Sources` panel
2. Add this source:

```text
rerun+http://127.0.0.1:19091/proxy
```

After that, the recording tree and entities will appear.

On newer `rerun-sdk` builds this uses `serve_grpc()` plus `serve_web_viewer()`.

Third-person camera support:

1. Enable it when creating the session (for example MCP `hab_init(..., third_person=true)` or
   `init_scene` payload with `sensor.third_person_color_sensor=true`).
2. `rerun_nav_viewer.py` will auto-export all available visual sensors and log them under:
   - `world/rgb/<sensor_name>`
   - `world/depth/<sensor_name>`
3. Third-person RGB is also mirrored at `world/rgb/third_person` for quick access.

You can also point to a different config file:

```bash
python3 tools/vis/rerun_nav_viewer.py --config /path/to/config.yaml
```

Only use `--spawn` on a machine with a local display.

## Recommended usage in proxy-enabled shells

```bash
NO_PROXY=127.0.0.1,localhost python3 tools/vis/rerun_nav_viewer.py --loop-id <loop_id> --serve-web 9091 --grpc-port 19091
```

## Watching multiple nav loops

Run one viewer per loop, using different ports:

```bash
# Loop A
python3 tools/vis/rerun_nav_viewer.py --loop-id <loop_a> --serve-web 9091 --grpc-port 19091

# Loop B (in a second terminal)
python3 tools/vis/rerun_nav_viewer.py --loop-id <loop_b> --serve-web 9092 --grpc-port 19092
```

## Quick Diagnostics

| Symptom | Check |
|---------|-------|
| Viewer shows no data | Confirm `--bridge-port` matches the bridge the nav loop is connected to |
| Images frozen / stale | Session may be idle; check nav loop status via `get_nav_loop_status` |
| Connection refused | Bridge not running — start with `python3 tools/habitat_agent.py --bridge-only` |
| Proxy error in viewer | Set `NO_PROXY=127.0.0.1,localhost` before launching the viewer |
| Blueprint not applied | Upgrade `rerun-sdk` to ≥0.16 or use `--spawn` to open the desktop viewer |
