import re
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from io import BytesIO
from itertools import product
from typing import Tuple

import vtracer
from PIL import Image

from ..utils import compare_pil_images, optimize_svg_with_scour, svg_to_png

from ..common import registry
from .base import IConverter


@registry.register_converter("vtracer-grid-search")
class VtracerGribSearch(IConverter):
    def __init__(self):
        super().__init__()

        self.default_svg = """<svg width="256" height="256" viewBox="0 0 256 256"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""
        self.speckle_values = [10, 20, 40]
        self.layer_diff_values = [64, 128]
        self.color_precision_values = [4, 5, 6]

    def _resize_image(self, image, size: Tuple[int, int] = (256, 256)) -> Image.Image:
        resized_image = image.resize(size, Image.Resampling.LANCZOS)
        return resized_image

    def _remove_version_attribute(self, svg_str: str) -> str:
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        tree = ET.ElementTree(ET.fromstring(svg_str))
        root = tree.getroot()

        if "version" in root.attrib:
            del root.attrib["version"]

        output = BytesIO()
        tree.write(output, encoding="utf-8", xml_declaration=True)
        return output.getvalue().decode("utf-8")

    def _remove_xml_tag(self, svg_str: str) -> str:
        return re.sub(r"<\?xml[^>]+\?>\s*", "", svg_str)

    def _convert_by_vtracer(self, image, limit=10000) -> str:
        best_svg = ""
        best_ssim = 0.0
        best_size = 0

        rgba_image = image.convert("RGBA")
        resized_image = self._resize_image(rgba_image)
        pixels = list(resized_image.getdata())

        for filter_speckle, layer_difference, color_precision in product(self.speckle_values, self.layer_diff_values, self.color_precision_values):
            svg_code = vtracer.convert_pixels_to_svg(
                rgba_pixels=pixels,
                size=resized_image.size,
                colormode="color",        # ["color"] or "binary"
                hierarchical="stacked",     # ["stacked"] or "cutout"
                mode="polygon",             # ["spline"], "polygon", "none"
                filter_speckle=filter_speckle,   # default: 4
                color_precision=color_precision,  # default: 6
                layer_difference=layer_difference,  # default: 16
                corner_threshold=60,  # default: 60
                length_threshold=4.0,  # in [3.5, 10] default: 4.0
                max_iterations=10,   # default: 10
                splice_threshold=45,  # default: 45
                path_precision=8,   # default: 8
            )

            compressed_svg = optimize_svg_with_scour(svg_code)
            byte_len = len(compressed_svg.encode('utf-8'))

            ssim = compare_pil_images(image, svg_to_png(compressed_svg))

            if byte_len <= limit and byte_len > best_size:
                if best_ssim <= ssim:
                    best_ssim = ssim
                    best_svg = self._remove_version_attribute(compressed_svg)
                    best_svg = self._remove_xml_tag(best_svg)
                    best_size = byte_len

        if best_svg:
            return best_svg
        else:
            return self.default_svg

    def convert_all(self, images: list[Image.Image], max_workers=4, limit=10000) -> list[str]:

        svgs = []
        convert_func = partial(self._convert_by_vtracer, limit=limit)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, svg in enumerate(executor.map(convert_func, images), 1):
                svgs.append(svg)
                # Log progress every 10 images or at the end
        return svgs

    def process(self, images, limit=20000):
        return self.convert_all(images, limit=limit)
