"""Capture one RGB565 frame from the INT8 ESP32-CAM serial firmware."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import serial


BEGIN_MARKER = b"camera_jpeg_begin:"
END_MARKER = b"\ncamera_jpeg_end\r\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM8")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/camera_captures/camera_frame.png"),
    )
    parser.add_argument("--timeout", type=float, default=45.0)
    return parser.parse_args()


def read_until(port: serial.Serial, marker: bytes, deadline: float) -> bytearray:
    buffer = bytearray()
    while time.monotonic() < deadline:
        buffer.extend(port.read(1))
        if marker in buffer:
            return buffer
        if len(buffer) > len(marker):
            del buffer[: -len(marker)]
    raise TimeoutError(f"Timed out waiting for {marker!r}.")


def read_exact(port: serial.Serial, size: int, deadline: float) -> bytes:
    frame = bytearray()
    while len(frame) < size and time.monotonic() < deadline:
        frame.extend(port.read(size - len(frame)))
    return bytes(frame)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with serial.Serial(args.port, 921600, timeout=0.2) as port:
        port.dtr = False
        port.rts = True
        time.sleep(0.1)
        port.rts = False
        deadline = time.monotonic() + args.timeout
        header = read_until(port, BEGIN_MARKER, deadline)
        length_digits = bytearray()
        while time.monotonic() < deadline:
            byte = port.read(1)
            if byte == b"\n":
                break
            length_digits.extend(byte)
        try:
            frame_length = int(length_digits)
        except ValueError as error:
            raise RuntimeError(f"Invalid JPEG length header: {length_digits!r}.") from error
        if not 100 <= frame_length <= 100000:
            raise RuntimeError(f"Invalid JPEG frame length: {frame_length}.")
        frame = read_exact(port, frame_length, deadline)
        if len(frame) != frame_length:
            raise RuntimeError(
                f"Expected {frame_length} JPEG bytes, received {len(frame)}."
            )
        suffix = port.read_until(END_MARKER)
        if suffix != END_MARKER:
            raise RuntimeError("Camera frame terminator was not received.")

    args.output.write_bytes(frame)
    print(f"saved={args.output}")
    print(f"frame_bytes={len(frame)}")
    print("format=JPEG received from ESP32 software conversion")


if __name__ == "__main__":
    main()
