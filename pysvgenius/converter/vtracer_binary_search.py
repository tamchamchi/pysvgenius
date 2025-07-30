import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import vtracer
from PIL import Image

from ..common import registry
from .base import IConverter


@registry.register_converter("vtracer-binary-search")
class VtracerBinarySearch(IConverter):
    def __init__(self):
        super().__init__()

    def _convert_svg_with_vtracer(
        self, img: Image.Image, image_size: tuple[int, int] = (256, 256)
    ) -> str:
        tmp_dir = tempfile.TemporaryDirectory()

        resized_image = img.resize(image_size)
        image_rgb = resized_image.convert("RGB")
        tmp_file_path = os.path.join(tmp_dir.name, "tmp.png")
        image_rgb.save(tmp_file_path)

        svg_path = os.path.join(tmp_dir.name, "converted_svg.svg")

        vtracer.convert_image_to_svg_py(
            image_path=tmp_file_path,
            out_path=svg_path,
            colormode="color",
            hierarchical="cutout",
            # hierarchical="stacked",
            mode="polygon",
        )

        with open(svg_path, "r", encoding="utf-8") as f:
            svg_str = f.read()
        return svg_str

    def _optimize_svg_path_vt(self, elem: str) -> str:
        """
        Takes a <path> element in M x,y L ... Z format,
        and returns the shortest equivalent path code that
        draws the same shape.
        Optimizes all "M…Z" subpaths in order.
        """
        # --- Extract attributes --------------------------------------------------
        d_match = re.search(r'd="([^"]+)"', elem)
        if not d_match:
            return elem
        d_raw = d_match.group(1)

        fill_m = re.search(r'fill="([^"]+)"', elem)
        fill = fill_m.group(1) if fill_m else None

        # --- Split by subpaths (M…Z blocks) -------------------------------------
        subpaths = re.findall(r'M[^M]*?Z', d_raw)

        optimized_subs = []
        for sub in subpaths:
            # --- Convert string to coordinate list -------------------------------
            tokens = re.findall(r'[MLZ]|-?\d+', sub)
            pts, cmd, i = [], None, 0
            while i < len(tokens):
                t = tokens[i]
                if t in ('M', 'L', 'Z'):
                    cmd, i = t, i + 1
                    continue
                if cmd in ('M', 'L'):
                    pts.append((int(t), int(tokens[i + 1])))
                    i += 2
                else:
                    i += 1

            # Skip this subpath if no valid coordinates
            if not pts:
                continue

            # --- Build the shortest path command sequence ------------------------
            d_parts = [f'M{pts[0][0]} {pts[0][1]}']
            prev_x, prev_y = pts[0]

            for x, y in pts[1:]:
                dx, dy = x - prev_x, y - prev_y

                cands = []
                # ― Absolute commands ―
                if dy == 0:
                    cands.append(f'H{x}')
                if dx == 0:
                    cands.append(f'V{y}')
                cands.append(f'L{x} {y}')
                # ― Relative commands ―
                if dy == 0:
                    cands.append(f'h{dx}')
                if dx == 0:
                    cands.append(f'v{dy}')
                cands.append(f'l{dx} {dy}')

                # Choose the shortest command
                d_parts.append(min(cands, key=len))
                prev_x, prev_y = x, y

            d_parts.append('Z')
            # Skip if only "M" and "Z"
            if len(d_parts) > 2:
                optimized_subs.append(''.join(d_parts))

        if not optimized_subs:
            return None

        d_optimized = ''.join(optimized_subs)
        return f'<path d="{d_optimized}"' + (f' fill="{fill}"' if fill else '') + '/>'

    def _remove_transform_from_path(self, path_text: str) -> str:
        """
        Takes an SVG <path> tag string and:
        - Removes transform="translate(tx,ty)"
        - Adds (tx, ty) to all coordinates in the `d` attribute
        Then returns the modified string.
        """
        # 1) Extract tx, ty from the transform attribute
        m_tx = re.search(
            r'transform=(["\'])\s*translate\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)\s*\1',
            path_text
        )
        if not m_tx:
            # If there's no transform, return as is
            return path_text

        tx, ty = float(m_tx.group(2)), float(m_tx.group(3))

        # 2) Extract the content of the `d` attribute
        m_d = re.search(r'd=(["\'])(?P<d>.*?)\1', path_text)
        if not m_d:
            # If `d` attribute not found, just remove transform and return
            return re.sub(r'\s+transform=(["\']).*?\1', '', path_text)

        d_orig = m_d.group('d')

        # 3) Find coordinate pairs “x,y” and shift them
        def shift_coord(m):
            x, y = float(m.group(1)), float(m.group(2))
            x2, y2 = x + tx, y + ty

            def fmt(v: float) -> str:
                return str(int(v)) if v.is_integer() else ('%f' % v).rstrip('0').rstrip('.')

            return fmt(x2) + ',' + fmt(y2)

        d_shifted = re.sub(
            r'([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)', shift_coord, d_orig)

        # 4) Replace the old `d` attribute with the new shifted one, and remove transform
        start, end = m_d.span()
        before_d = path_text[:start]
        after_d = path_text[end:]
        new_path = before_d + f'd="{d_shifted}"' + after_d
        new_path = re.sub(r'\s+transform=(["\']).*?\1', '', new_path)

        return new_path

    def _extract_paths(self, svg):
        """
        Extracts all <path> opening tags from an SVG string,
        removes transform attributes, and returns a list
        of cleaned path tags in their original order.
        """
        # 1.Extract only <path> tags in order
        tag_pattern = re.compile(r'<path\b[^>]*>', flags=re.IGNORECASE)
        raw_tags = [m.group(0) for m in tag_pattern.finditer(svg)]

        # 2.Clean each path tag string
        return [self._remove_transform_from_path(t) for t in raw_tags]

    def _extract_svg_size_with_regex(self, svg):
        """Extract width and height from an SVG tag using regular expressions"""
        w_match = re.search(r'<svg[^>]*\bwidth=["\']([^"\']+)["\']', svg)
        h_match = re.search(r'<svg[^>]*\bheight=["\']([^"\']+)["\']', svg)
        width = w_match.group(1) if w_match else None
        height = h_match.group(1) if h_match else None
        return width, height

    def _svg_compress(self, svg: str) -> str:
        """
        Compresses an SVG string by:
        - Extracting all paths
        - Optimizing each path to its shortest form
        - Rebuilding a compact SVG with consistent header and footer
        """
        path_list = self._extract_paths(svg)
        path_list = [self._optimize_svg_path_vt(p) for p in path_list]
        path_list = [p for p in path_list if p is not None]

        width, height = self._extract_svg_size_with_regex(svg)
        header = f'<svg xmlns="http://www.w3.org/2000/svg" width="384" height="384" viewBox="0 0 {width} {height}">'
        fill_background = re.search(r'fill="([^"]+)"', path_list[0])
        header += f'<rect width="{width}" height="{height}" fill="{fill_background.group(1)}"/>'

        # footer = f'<path d="M167 200h20l-20 30h20" fill="none" stroke="white" stroke-width="{4*int(width)/256}" transform="scale({int(width)/256} {int(height)/256})" stroke-linecap="round" stroke-linejoin="round"/><path d="M167 200h20l-20 30h20" fill="none" stroke="#EDCE6A" stroke-width="{2*int(width)/256}" transform="scale({int(width)/256} {int(height)/256})" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        footer = "</svg>"
        return header + ''.join(path_list[1:]) + footer

    def vtracer_png_to_svg(self, image, size_range=(50, 500), limit=10000):
        """
        Convert the given image to an SVG and compress it,
        using binary search to find the largest size such that
        the length of the resulting SVG string does not exceed `limit`.

        Parameters
        ----------
        image : any
            Input image object to be vectorized into SVG.
        size_range : tuple(int, int)
            Minimum and maximum image sizes (inclusive) to search over.
        limit : int
            Maximum allowed length (in characters) for the SVG string.

        Returns
        -------
        best_size : int
            The largest image size for which len(svg) <= limit.
        best_svg : str
            The SVG string converted and compressed at best_size.

        Raises
        ------
        ValueError
            If no image size within `size_range` can satisfy the length limit.
        """
        lo, hi = size_range
        best_svg = None

        # Binary search to find the largest size that keeps SVG under limit
        while abs(lo - hi) > 5:
            mid = (lo + hi) // 2

            # Convert to SVG and compress
            svg = self._convert_svg_with_vtracer(image, (mid, mid))
            svg = self._svg_compress(svg)
            length = len(svg)

            if length <= limit:
                # This size is valid → Try larger sizes
                best_svg = svg
                lo = mid + 1
            else:
                # SVG too large → Try smaller sizes
                hi = mid - 1

        # Final check: if we still don't have a valid SVG, try the minimum size
        if best_svg is None:
            svg = self._convert_svg_with_vtracer(image, (lo, lo))
            svg = self._svg_compress(svg)
            if len(svg) <= limit:
                best_svg = svg
            else:
                # Return a minimal SVG as fallback
                best_svg = f'<svg width="{lo}" height="{lo}" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="gray"/></svg>'

        return best_svg

    def convert_all(self, images: list[Image.Image], max_workers=4, limit=10000) -> list[str]:
        svgs = []
        convert_func = partial(self.vtracer_png_to_svg, limit=limit)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, svg in enumerate(executor.map(convert_func, images), 1):
                svgs.append(svg)

        if len(svgs) == 1:
            return svgs[0]

        return svgs

    def process(self, images: Image.Image | list[Image.Image], limit=10000) -> list[str]:
        if isinstance(images, Image.Image):
            images = [images]
        svgs = self.convert_all(images=images, max_workers=4, limit=limit)
        return svgs
