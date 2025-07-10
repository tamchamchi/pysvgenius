from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import tempfile
import re

import vtracer
from .base import IImageToConverter
from PIL import Image
import os

from utils.image_utils import svg_to_png


class VtracerConverter(IImageToConverter):
    def __init__(self):
        self.limit = 10000

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
        header = f'<svg width="384" height="384" viewBox="0 0 {width} {height}">'
        fill_background = re.search(r'fill="([^"]+)"', path_list[0])
        header += f'<rect width="{width}" height="{height}" fill="{fill_background.group(1)}"/>'

        # footer = f'<path d="M167 200h20l-20 30h20" fill="none" stroke="white" stroke-width="{4*int(width)/256}" transform="scale({int(width)/256} {int(height)/256})" stroke-linecap="round" stroke-linejoin="round"/><path d="M167 200h20l-20 30h20" fill="none" stroke="#EDCE6A" stroke-width="{2*int(width)/256}" transform="scale({int(width)/256} {int(height)/256})" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        footer = "</svg>"
        return header + ''.join(path_list[1:]) + footer

    def _svg_conversion_division(self, img: Image.Image, image_size=(384, 384), num_divisions=5):
        """
        1. Resize the image based on image_size and divide it into num_divisions x num_divisions tiles.
        2. For each tile, crop the original image accordingly and convert it into SVG.
        3. Finally, merge all the tile SVG paths, adjusting coordinates directly instead of using transforms.
        """
        # Resize the input image to the specified image_size
        img_resized = img.resize(image_size, Image.LANCZOS)
        W, H = img_resized.size

        # Calculate the base size for each tile
        base_tile_w = W // num_divisions
        base_tile_h = H // num_divisions

        if base_tile_w == 0 or base_tile_h == 0:
            # If the resulting tile size becomes 0, issue a warning and continue with minimal size
            print(
                f"Warning: Calculated base tile size is very small ({base_tile_w}x{base_tile_h}). "
                "Resulting SVG might be distorted or empty."
            )
            # Set minimum size to 1 to continue processing
            base_tile_w = max(1, base_tile_w)
            base_tile_h = max(1, base_tile_h)

        all_shifted_path_tags = []
        overall_bg_color = None  # Background color of the entire SVG

        for r_idx in range(num_divisions):
            for c_idx in range(num_divisions):
                offset_x = c_idx * base_tile_w
                offset_y = r_idx * base_tile_h

                # Calculate the actual width and height for the current tile (last tiles may include remainders)
                current_tile_w = base_tile_w if c_idx < num_divisions - 1 else W - offset_x
                current_tile_h = base_tile_h if r_idx < num_divisions - 1 else H - offset_y

                if current_tile_w <= 0 or current_tile_h <= 0:
                    continue  # Skip tiles with zero or negative size

                # Crop the tile from the resized image
                box = (offset_x, offset_y, offset_x +
                       current_tile_w, offset_y + current_tile_h)
                tile_img = img_resized.crop(box)

                if tile_img.width == 0 or tile_img.height == 0:
                    continue

                # Same logic as svg_conversion: convert tile to SVG using vtracer
                try:
                    with tempfile.TemporaryDirectory() as tmp_dir_name:
                        tmp_file_path = os.path.join(
                            tmp_dir_name, f"tmp_tile_{r_idx}_{c_idx}.png")
                        # Save in RGB mode
                        tile_img.convert("RGB").save(tmp_file_path)

                        svg_tile_output_path = os.path.join(
                            tmp_dir_name, f"tile_gen_svg_{r_idx}_{c_idx}.svg")
                        vtracer.convert_image_to_svg_py(
                            tmp_file_path,
                            svg_tile_output_path,
                            colormode="color",
                            hierarchical="cutout",  # Use cutout mode
                            mode="polygon",
                            # Add other vtracer parameters here if needed
                        )
                        with open(svg_tile_output_path, encoding="utf-8") as f:
                            tile_svg_str = f.read()
                except Exception as e:
                    print(f"Error processing tile ({r_idx},{c_idx}): {e}")
                    continue

                # Extract <path> elements from the tile SVG
                raw_path_tags_from_tile = self._extract_paths(tile_svg_str)

                if not raw_path_tags_from_tile:
                    continue

                # Try to get the background color from the first path of the first tile
                if r_idx == 0 and c_idx == 0 and not overall_bg_color:
                    first_path_of_first_tile = raw_path_tags_from_tile[0]
                    match_bg = re.search(
                        r'fill="([^"]+)"', first_path_of_first_tile)
                    if match_bg:
                        overall_bg_color = match_bg.group(1)

                # Offset the coordinates of each path
                for path_idx, path_tag_original in enumerate(raw_path_tags_from_tile):
                    d_match = re.search(r'd="([^"]+)"', path_tag_original)
                    if not d_match:
                        continue
                    d_original = d_match.group(1)

                    shifted_sub_paths_strings = []
                    # Split into subpaths using "M...Z" pattern
                    sub_paths_data = re.findall(
                        r"M[^M]*?Z", d_original, flags=re.IGNORECASE)
                    if not sub_paths_data and d_original.strip():  # If no subpaths, treat as one whole path
                        sub_paths_data = [d_original]

                    valid_path_data_found = False
                    for sub_d_segment in sub_paths_data:
                        # Tokens are commands (letters) or numbers
                        tokens = re.findall(
                            r"[MLZHVCSQTA]|-?[\d\.]+", sub_d_segment, flags=re.IGNORECASE)
                        shifted_tokens_for_sub = []
                        k = 0
                        while k < len(tokens):
                            cmd = tokens[k]
                            shifted_tokens_for_sub.append(cmd)
                            k += 1

                            # Handle coordinates based on the number of arguments for each command
                            # vtracer polygon mode is expected to use mainly M and L (absolute coordinates)
                            # Assume coordinates are integers based on vtracer polygon output
                            coords_to_process = 0
                            if cmd.upper() in ("M", "L", "T"):
                                coords_to_process = 1  # One x,y pair
                            elif cmd.upper() in ("Q", "S"):
                                coords_to_process = 2  # Two x,y pairs
                            elif cmd.upper() == "C":
                                coords_to_process = 3  # Three x,y pairs
                            # A (Arc) is complex and assumed to be unused in polygon mode
                            # H, V are single coordinates

                            if cmd.upper() == "H":  # Horizontal line
                                if k < len(tokens):
                                    try:
                                        x = int(float(tokens[k]))
                                        shifted_tokens_for_sub.append(
                                            str(x + offset_x))
                                        k += 1
                                    except (ValueError, IndexError):
                                        break  # Parse error
                            elif cmd.upper() == "V":  # Vertical line
                                if k < len(tokens):
                                    try:
                                        y = int(float(tokens[k]))
                                        shifted_tokens_for_sub.append(
                                            str(y + offset_y))
                                        k += 1
                                    except (ValueError, IndexError):
                                        break  # Parse error
                            else:  # Commands with x,y pairs
                                for _ in range(coords_to_process):
                                    if k + 1 < len(tokens):
                                        try:
                                            x = int(float(tokens[k]))
                                            y = int(float(tokens[k + 1]))
                                            shifted_tokens_for_sub.append(
                                                str(x + offset_x))
                                            shifted_tokens_for_sub.append(
                                                str(y + offset_y))
                                            k += 2
                                        except (ValueError, IndexError):
                                            # Exit loop on error
                                            k = len(tokens)
                                            break
                                    else:  # Not enough tokens
                                        k = len(tokens)
                                        break
                                # Ended prematurely
                                if k == len(tokens) and _ < coords_to_process - 1:
                                    # Only keep the command
                                    shifted_tokens_for_sub = [
                                        shifted_tokens_for_sub[0]]

                        if len(shifted_tokens_for_sub) > 1:  # Command + at least one argument
                            shifted_sub_paths_strings.append(
                                " ".join(shifted_tokens_for_sub))
                            valid_path_data_found = True

                    if valid_path_data_found:
                        shifted_d_final = "".join(shifted_sub_paths_strings)
                        # Replace only the d attribute in the original path tag
                        new_path_tag = re.sub(
                            r'd="[^"]*"', f'd="{shifted_d_final}"', path_tag_original, count=1)
                        all_shifted_path_tags.append(new_path_tag)

        # Construct overall SVG header using resized width and height
        svg_final_header = f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">'

        # Concatenate all adjusted paths
        final_svg_str = svg_final_header + \
            "".join(all_shifted_path_tags) + "</svg>"

        return final_svg_str

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
        best_size = None
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
                best_size = mid
                best_svg = svg
                lo = mid + 1
            else:
                # SVG too large → Try smaller sizes
                hi = mid - 1

        # If size is small, try subdividing the image and recompress
        if best_size < 256:
            return best_svg
        else:
            for i in range(2, 14):
                svg = self._svg_conversion_division(image, (256, 256), i)
                svg = self._svg_compress(svg)
                length = len(svg)
                if length < limit:
                    best_svg = svg
                else:
                    break

        return best_svg

    def convert_all(self, images: list[Image.Image], max_workers=4, limit=10000) -> list[str]:
        svgs = []
        convert_func = partial(self.vtracer_png_to_svg, limit=limit)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for svg in tqdm(executor.map(convert_func, images), total=len(images), desc="Converting SVGs..."):
                svgs.append(svg)
        return svgs

    def process(self, images: list[Image.Image], limit=10000) -> list[str]:
        svgs = self.convert_all(images=images, max_workers=4, limit=limit)
        return svgs


if __name__ == "__main__":
    convertor = VtracerConverter()
    image = Image.open(
        "/home/anhndt/pysvgenius/data/test/raw_image.png")
    svg = convertor.process([image], limit=10000)
    converted_image = svg_to_png(svg[0])
    converted_image.save("image.png")
    with open("test_svg.svg", 'w') as f:
        f.write(svg[0])
    print(len(svg[0].encode('utf-8')))
