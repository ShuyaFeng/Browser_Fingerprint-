"""Shared figure style: map raw dataset column names to the reader-facing
feature names used in the paper text, so figures and prose agree."""

LABELS = {
    # Li & Cao corpus columns
    "jsFonts": "Fonts",
    "fp2_webgl": "WebGL hash",
    "gpu": "GPU renderer",
    "agent": "User-agent",
    "hybridaudio": "Audio",
    "canvastest": "Canvas",
    "language": "Language",
    "fp2_pixelratio": "Pixel ratio",
    "osversion": "OS version",
    "cpucores": "CPU cores",
    "timezone": "Time zone",
    "doNotTrack": "Do-not-track",
    "browserversion": "Browser version",
    "fp2_platform": "Platform",
    "browser": "Browser",
    "os": "OS",
    "encoding": "Accept-encoding",
    "fp2_colordepth": "Color depth",
    # FPStalker-side columns (note: 'fp2_webglvendoe' is the raw column name)
    "resolution": "Resolution",
    "fp2_webglvendoe": "WebGL vendor",
}


def label(feature):
    """Reader-facing label for a raw feature column name."""
    return LABELS.get(feature, feature)


def labels(features):
    return [label(f) for f in features]
