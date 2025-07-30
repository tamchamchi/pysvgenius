from scour import scour
import re
from collections import defaultdict


def optimize_svg_with_scour(svg):
    options = scour.parse_args([
        '--enable-viewboxing',
        '--enable-id-stripping',
        '--enable-comment-stripping',
        '--shorten-ids',
        '--indent=none',
        '--strip-xml-prolog',
        '--remove-metadata',
        '--remove-descriptive-elements',
        '--disable-embed-rasters',
        '--enable-viewboxing',
        '--create-groups',
        '--renderer-workaround',
        '--set-precision=2',
    ])

    svg = scour.scourString(svg, options)

    svg = svg.replace('id=""', '')
    svg = svg.replace('version="1.0"', '')
    svg = svg.replace('version="1.1"', '')
    svg = svg.replace('version="2.0"', '')
    svg = svg.replace('  ', ' ')
    svg = svg.replace('>\n', '>')

    return svg


def extract_rect_and_path(svg_text, resize):
    """
    Extract opening <rect> and <path> tags from SVG string,
    removing opacity, converting RGB to hex, and rounding values.

    Returns:
        A list of processed tag strings in order of appearance.
    """
    tag_pattern = re.compile(r'<(?:rect|path)\b[^>]*>', flags=re.IGNORECASE)
    raw_tags = [m.group(0) for m in tag_pattern.finditer(svg_text)]

    return [_process_tag(t, resize) for t in raw_tags]


def _process_tag(tag, resize):
    # 1) Remove `opacity` attributes
    tag = re.sub(r'\s*opacity\s*=\s*"[^"]*"', '', tag, flags=re.IGNORECASE)

    # 2) Convert fill="rgb(R,G,B)" to fill="#RRGGBB"
    def _rgb_to_hex(m: re.Match) -> str:
        nums = [float(m.group(i)) for i in (1, 2, 3)]
        ints = [int(round(v)) for v in nums]
        return f'fill="#{ints[0]:02X}{ints[1]:02X}{ints[2]:02X}"'

    tag = re.sub(
        r'fill\s*=\s*"rgb\(\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)"',
        _rgb_to_hex,
        tag,
        flags=re.IGNORECASE
    )

    # 3) Round all decimal numbers and scale by resize factor
    def _round_num(m: re.Match) -> str:
        return str(int(round(float(m.group(0)) * resize / 384)))

    tag = re.sub(r'-?\d+\.\d+', _round_num, tag)
    return tag


def _close_path_with_Z(d: str) -> str:
    """
    If the path string ends with 'L x y', replace it with 'Z'.
    Otherwise, return the original path.
    """
    m = re.match(
        r'^(.*?)(?:\s+L\s+[-+]?\d*\.?\d+(?:[,\s]+[-+]?\d*\.?\d+)\s*)$', d)
    if m:
        return m.group(1) + ' Z'
    return d


def merge_paths_by_fill(paths: list[str]) -> list[str]:
    """
    Merge <path> elements with the same fill color.
    - Concatenate `d` attributes.
    - Close each subpath with 'Z' if necessary.

    Returns:
        List of <path> elements with merged `d` and same fill.
    """
    paths_by_fill = defaultdict(list)

    for path_str in paths:
        # Extract `d` attribute
        d_match = re.search(r'd\s*=\s*["\']([^"\']+)["\']', path_str)
        if not d_match:
            continue
        d = d_match.group(1)

        # Extract fill attribute
        fill_match = re.search(r'fill\s*=\s*["\']([^"\']*)["\']', path_str)
        fill = fill_match.group(1) if fill_match else ''

        # Fix closing
        d_fixed = _close_path_with_Z(d)
        paths_by_fill[fill].append(d_fixed)

    # Merge each group
    merged_paths = []
    for fill, d_list in paths_by_fill.items():
        merged_d = " ".join(d_list)
        fill_attr = f' fill="{fill}"' if fill else ''
        merged_path = f'<path d="{merged_d}"{fill_attr} />'
        merged_paths.append(merged_path)

    return merged_paths


def _shortest_segment(prev, cur):
    """
    Return the shortest SVG command for moving from `prev` to `cur`.
    Uses relative and absolute commands and chooses the shortest.
    """
    px, py = prev
    x, y = cur
    dx, dy = x - px, y - py
    cands = [
        (f'H{x}', abs(dy) == 0),           # Absolute horizontal
        (f'h{dx}', abs(dy) == 0),           # Relative horizontal
        (f'V{y}', abs(dx) == 0),           # Absolute vertical
        (f'v{dy}', abs(dx) == 0),           # Relative vertical
        (f'L{x} {y}', True),                 # Absolute diagonal
        (f'l{dx} {dy}', True),                 # Relative diagonal
    ]
    # Filter by condition and select the shortest string
    return min((s for s, ok in cands if ok), key=len)


def _encode_path(pts):
    """List of points → shortest SVG command sequence (with Z)"""
    d = [f'M{pts[0][0]} {pts[0][1]}']
    for a, b in zip(pts, pts[1:]):
        d.append(_shortest_segment(a, b))
    d.append('')
    return ''.join(d)


def svg_path_extream(elem: str) -> str | None:
    """
    Given a <path> element (possibly multiple M…Z subpaths),
    - Try all starting points (cyclic permutation)
    - Try both clockwise and counter-clockwise
    and return the shortest possible path `d`.

    Returns None if no valid path is found.
    """
    # --- Extract attributes -------------------------------------------------
    m_d = re.search(r'd="([^"]+)"', elem)
    d_raw = m_d.group(1)

    fill_m = re.search(r'fill="([^"]+)"', elem)
    fill = f' fill="{fill_m.group(1)}"'

    # --- Extract subpaths (M…Z) ---------------------------------------------
    subs_raw = re.findall(r'M[^M]*?Z', d_raw)
    optimized = []

    for sub in subs_raw:
        # Tokenize to points
        toks = sub.replace(',', ' ').split()
        pts, cmd, i = [], None, 0
        while i < len(toks):
            t = toks[i]
            if t in ('M', 'L', 'Z'):
                cmd, i = t, i + 1
            elif cmd in ('M', 'L'):
                pt = (int(float(t)), int(float(toks[i + 1])))
                if len(pts) == 0 or pts[-1] != pt:
                    pts.append(pt)
                i += 2
            else:
                i += 1
        if len(pts) < 3:  # Skip if not enough points for a shape
            continue

        # --- Try all N×2 permutations (start point × direction) --------------
        best, best_len = sub, len(sub)
        for seq in (pts, pts[::-1]):  # Normal and reversed
            n = len(seq)
            for k in range(n):
                rot = seq[k:] + seq[:k]  # Rotate starting point
                d_candidate = _encode_path(rot)
                if (l := len(d_candidate)) < best_len:
                    best, best_len = d_candidate, l

        optimized.append(best)

    if not optimized:  # No valid shapes
        return None

    d_final = ''.join(optimized)
    return f'<path d="{d_final}"{fill}/>'


def optimize_svg_size(svg, resize=384, limit=10000):
    """
    Resize and optimize an SVG for a fixed canvas.
    If `limit` is True, it trims the <path> elements to fit within 10KB.

    Returns an optimized SVG string.
    """
    canvas_width = resize
    canvas_height = resize
    header = f'<svg xmlns="http://www.w3.org/2000/svg" width="384" height="384" viewBox="0 0 {canvas_width} {canvas_height}">'
    # footer = f'<g fill="none" transform="scale({round(canvas_width/256,2)} {round(canvas_height/256,2)})" stroke-linecap="round" stroke-linejoin="round"><path d="M167 200h20l-20 30h20" stroke="#fff" stroke-width="4"/><path d="M167 200h20l-20 30h20" stroke="#EDCE6A" stroke-width="2"/></g></svg>'
    footer = "</svg>"
    # Ignore text layers (last 2 elements)
    svg_opt_list = extract_rect_and_path(svg, resize)[:-2]

    # Split background <rect> and <path> elements
    rect = svg_opt_list[0]
    rect = re.sub(r'\s*(?:x|y)="0"', '', rect)
    header += rect

    path_list = svg_opt_list[1:]
    # Merge paths with same fill color
    path_list = merge_paths_by_fill(path_list)

    # Compress and optimize paths
    for i in range(len(path_list)):
        path_list[i] = svg_path_extream(path_list[i])

    # Remove any None results (invalid paths)
    path_list = [p for p in path_list if p is not None]

    svg_opt = header + ''.join(path_list) + footer

    # Trim to 10KB if limit is True
    if limit:
        len_tmp = len((header + footer).encode())
        idx = 0
        for i in range(len(path_list)):
            len_tmp += len(path_list[i].encode())
            if len_tmp > limit:
                break
            idx += 1
        svg_opt = header + "".join(path_list[:idx]) + footer

    return svg_opt
