import base64
import struct
import zlib

import numpy as np

from hsim_manipulation.vla_client import png_b64, resize_nearest


def test_png_roundtrip():
    img = (np.random.default_rng(0).random((24, 32, 3)) * 255).astype(np.uint8)
    png = base64.b64decode(png_b64(img))
    assert png[:8] == b'\x89PNG\r\n\x1a\n'
    w, h = struct.unpack('>II', png[16:24])
    assert (w, h) == (32, 24)
    idat = png[png.index(b'IDAT') + 4: png.index(b'IEND') - 8]
    raw = zlib.decompress(idat)
    rows = np.frombuffer(raw, np.uint8).reshape(24, 1 + 32 * 3)
    assert (rows[:, 1:].reshape(24, 32, 3) == img).all()


def test_resize():
    img = np.zeros((480, 640, 3), np.uint8)
    assert resize_nearest(img, 320, 240).shape == (240, 320, 3)
