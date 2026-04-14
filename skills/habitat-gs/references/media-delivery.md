# Media Delivery

How to send media files (images, videos) to users across different platforms.

## General principle

The agent should use the platform's `message` tool with a `media` parameter pointing to the local file path. The platform plugin handles upload and delivery automatically.

**Do NOT send file paths as plain text.** Always use the `media` parameter.

## Feishu (飞书)

### Sending a video

```
message(action="send", message="Navigation replay video", media="/home/node/.openclaw/workspace/artifacts/habitat-gs/<session_id>_steps000000-000NNN_color_sensor.mp4", filename="navigation_replay.mp4")
```

### Sending an image

```
message(action="send", message="Current camera view", media="/home/node/.openclaw/workspace/artifacts/habitat-gs/<session_id>_stepNNNNNN_color_sensor.png", filename="camera_view.png")
```

### Parameters

| Parameter | Required | Description |
|---|---|---|
| `action` | yes | `"send"` |
| `message` | yes | Text caption (sent as a separate message before the media) |
| `media` | yes | Absolute file path inside the container |
| `filename` | no | Display name for the file in Feishu |

### How it works

1. The Feishu plugin reads the local file from the container filesystem
2. Uploads it to Feishu cloud storage via `im/v1/files` API
3. Sends a file message with the obtained `file_key`
4. If upload fails, falls back to sending the file path as a clickable text link

### Common issues

| Issue | Cause | Fix |
|---|---|---|
| File path sent as text | Missing `media` parameter | Always use `media=<path>` in message tool |
| Upload failed | File not found in container | Verify path exists: `exec(command="ls -l <path>")` |
| No video frames | Bridge restarted mid-session | Re-init scene and re-run navigation |

## Telegram

(Placeholder for future Telegram integration)

The same `message` tool with `media` parameter should work. Platform-specific details to be added when Telegram channel is configured.

## Best practice

1. Always `export-video` before sending — ensures the mp4 file exists
2. Use the exact path returned by `export-video` in the `media` parameter
3. Include a brief text description in `message` (what the video shows)
4. If sending fails, check file existence with `exec(command="ls -l <path>")`
