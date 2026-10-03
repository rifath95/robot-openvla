"""Render one RGB observation from the fixed MuJoCo workspace camera."""

from pathlib import Path
import struct
import zlib

import mujoco


SCENE_PATH = Path(__file__).with_name("scene.xml")
OUTPUT_PATH = Path(__file__).with_name("camera_image.png")
WIDTH = 640
HEIGHT = 480


def save_rgb_png(path: Path, image) -> None:
    """Save an HxWx3 uint8 RGB array using only Python's standard library."""
    signature = b"\x89PNG\r\n\x1a\n"

    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", image.shape[1], image.shape[0], 8, 2, 0, 0, 0)
    scanlines = b"".join(b"\x00" + row.tobytes() for row in image)
    path.write_bytes(
        signature
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "scene_home")
    mujoco.mj_resetDataKeyframe(model, data, home_key)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        renderer.update_scene(data, camera="workspace_camera")
        rgb = renderer.render()
    save_rgb_png(OUTPUT_PATH, rgb)
    print(f"Saved {rgb.shape[1]}x{rgb.shape[0]} RGB image to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
