"""Show the ESP32-CAM livestream and classify LED health on the laptop."""

from __future__ import annotations

import argparse
import io
import time
from dataclasses import dataclass
from typing import Iterator
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import streamlit as st
from PIL import Image


MIN_VISIBLE_FRACTION = 0.01
STREAM_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class ColorPolicyResult:
    label: str
    color: str
    green_fraction: float
    yellow_fraction: float
    blue_fraction: float
    visible_pixels: int


def classify_frame(image: Image.Image) -> ColorPolicyResult:
    """Apply the same documented color policy used by the ESP32 page."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    red = rgb[..., 0].astype(np.float32)
    green = rgb[..., 1].astype(np.float32)
    blue = rgb[..., 2].astype(np.float32)
    visible = (red + green + blue) > 90
    blue_mask = (
        (blue > 80)
        & (blue > red * 1.2)
        & (blue > green * 1.1)
        & (blue - red > 20)
    )
    green_mask = (
        (green > 80)
        & (green > red * 1.15)
        & (green > blue * 1.15)
        & (green - red > 20)
    )
    yellow_mask = (
        (red > 80)
        & (green > 80)
        & (red > blue * 1.2)
        & (green > blue * 1.2)
        & (np.abs(red - green) < 90)
    )
    visible_count = int(np.count_nonzero(visible))
    denominator = max(visible_count, 1)
    fractions = {
        "blue": float(np.count_nonzero(blue_mask & visible) / denominator),
        "green": float(np.count_nonzero(green_mask & visible) / denominator),
        "yellow": float(np.count_nonzero(yellow_mask & visible) / denominator),
    }

    if (
        fractions["blue"] >= MIN_VISIBLE_FRACTION
        and fractions["blue"] >= fractions["green"]
        and fractions["blue"] >= fractions["yellow"]
    ):
        label, color = "BLUE LOCATE INDICATOR - HEALTH NOT CLASSIFIED", "blue"
    elif (
        fractions["green"] >= MIN_VISIBLE_FRACTION
        and fractions["green"] >= fractions["yellow"]
    ):
        label, color = "SERVER STATUS: WORKING / GREEN", "green"
    elif fractions["yellow"] >= MIN_VISIBLE_FRACTION:
        label, color = "SERVER STATUS: FAULT / YELLOW", "darkgoldenrod"
    else:
        label, color = "UNKNOWN / INSUFFICIENT COLOR SIGNAL", "gray"

    return ColorPolicyResult(
        label=label,
        color=color,
        green_fraction=fractions["green"],
        yellow_fraction=fractions["yellow"],
        blue_fraction=fractions["blue"],
        visible_pixels=visible_count,
    )


def jpeg_frames(stream_url: str) -> Iterator[Image.Image]:
    request = Request(stream_url, headers={"Cache-Control": "no-cache"})
    with urlopen(request, timeout=STREAM_TIMEOUT_SECONDS) as response:
        buffer = bytearray()
        while True:
            chunk = response.read(4096)
            if not chunk:
                return
            buffer.extend(chunk)
            while True:
                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9", start + 2)
                if start < 0 or end < 0:
                    if len(buffer) > 1_000_000:
                        del buffer[:-2]
                    break
                payload = bytes(buffer[start : end + 2])
                del buffer[: end + 2]
                try:
                    image = Image.open(io.BytesIO(payload)).convert("RGB")
                except (OSError, ValueError):
                    continue
                yield image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-url", default="http://10.20.134.230/stream")
    return parser.parse_args()


def render_dashboard(stream_url: str) -> None:
    st.set_page_config(page_title="Rover server health", layout="wide")
    st.title("Rover ESP32-CAM server health")
    st.caption("Laptop-side continuous classification of the ESP32-CAM MJPEG stream")
    st.info(
        "LED color is a proxy for server status, not proof of actual server health. "
        "Blue is a locate/identify indicator and is not a health class."
    )
    st.code(stream_url, language="text")
    frame_slot = st.empty()
    status_slot = st.empty()
    metrics_slot = st.empty()

    try:
        for frame in jpeg_frames(stream_url):
            result = classify_frame(frame)
            frame_slot.image(frame, channels="RGB", use_container_width=True)
            status_slot.markdown(
                f"<h2 style='color:{result.color}'>{result.label}</h2>",
                unsafe_allow_html=True,
            )
            metrics_slot.write(
                {
                    "visible_pixels": result.visible_pixels,
                    "green_fraction": round(result.green_fraction, 4),
                    "yellow_fraction": round(result.yellow_fraction, 4),
                    "blue_fraction": round(result.blue_fraction, 4),
                }
            )
            time.sleep(0.1)
    except (OSError, URLError, TimeoutError, ValueError) as error:
        st.error(f"Stream connection/classification failed: {error}")
        st.info("Check that the ESP32 and laptop are on the same 2.4 GHz network.")


if __name__ == "__main__":
    arguments = parse_args()
    render_dashboard(arguments.stream_url)
