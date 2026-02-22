"""Generate simple PNG icons for the Chrome extension."""
import struct, zlib, os

def create_png(size):
    pixels = []
    cx, cy = size / 2, size / 2
    r = size * 0.42

    for y in range(size):
        row = []
        for x in range(size):
            dx = (x - cx) / r
            dy = (y - cy) / r
            in_shield = (dx**2 + (dy * 0.85)**2 < 1.0) and (dy < 0.7 or (abs(dx) < (1.0 - dy) * 0.8))
            if in_shield:
                nx = (x - cx) / r
                ny = (y - cy) / r
                in_bar = abs(nx) < 0.12 and -0.55 < ny < 0.2
                in_dot = abs(nx) < 0.12 and 0.35 < ny < 0.52
                if in_bar or in_dot:
                    row.extend([255, 255, 255, 255])
                else:
                    row.extend([99, 102, 241, 255])
            else:
                row.extend([0, 0, 0, 0])
        pixels.append(bytes(row))

    def make_chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)

    header = b'\x89PNG\r\n\x1a\n'
    ihdr = make_chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 6, 0, 0, 0))
    raw = b''
    for row in pixels:
        raw += b'\x00' + row
    idat = make_chunk(b'IDAT', zlib.compress(raw, 9))
    iend = make_chunk(b'IEND', b'')
    return header + ihdr + idat + iend

icon_dir = os.path.join(os.path.dirname(__file__), 'icons')
os.makedirs(icon_dir, exist_ok=True)
for s in [16, 48, 128]:
    data = create_png(s)
    path = os.path.join(icon_dir, f'icon{s}.png')
    with open(path, 'wb') as f:
        f.write(data)
    print(f'Created {path} ({len(data)} bytes)')
