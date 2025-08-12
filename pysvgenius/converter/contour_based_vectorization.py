from concurrent.futures import ProcessPoolExecutor
from functools import partial

import cv2
import numpy as np
from PIL import Image

from ..common import registry
from .base import IConverter


@registry.register_converter("contour-based")
class ContourBasedConvertor(IConverter):
    def __init__(self):
        super().__init__()

    def _compress_hex_color(self, hex_color):
        """Convert hex color to shortest possible representation"""
        r, g, b = int(hex_color[1:3], 16), int(
            hex_color[3:5], 16), int(hex_color[5:7], 16)
        if r % 17 == 0 and g % 17 == 0 and b % 17 == 0:
            return f"#{r // 17:x}{g // 17:x}{b // 17:x}"
        return hex_color

    def _extract_features_by_scale(self, img_np, num_colors=16):
        """
        Extract image features hierarchically by scale

        Args:
            img_np (np.ndarray): Input image
            num_colors (int): Number of colors to quantize

        Returns:
            list: Hierarchical features sorted by importance
        """
        # Convert to RGB if needed
        if len(img_np.shape) == 3 and img_np.shape[2] > 1:
            img_rgb = img_np
        else:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

        # Convert to grayscale for processing
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape

        # Perform color quantization
        pixels = img_rgb.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # Quantized image
        palette = centers.astype(np.uint8)
        quantized = palette[labels.flatten()].reshape(img_rgb.shape)

        # Hierarchical feature extraction
        hierarchical_features = []

        # Sort colors by frequency
        unique_labels, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-counts)
        sorted_colors = [palette[i] for i in sorted_indices]

        # Center point for importance calculations
        center_x, center_y = width / 2, height / 2

        for color in sorted_colors:
            # Create color mask
            color_mask = cv2.inRange(quantized, color, color)

            # Find contours
            contours, _ = cv2.findContours(
                color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # Convert RGB to compressed hex
            hex_color = self._compress_hex_color(
                f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}")

            color_features = []
            for contour in contours:
                # Skip tiny contours
                area = cv2.contourArea(contour)
                if area < 20:
                    continue

                # Calculate contour center
                m = cv2.moments(contour)
                if m["m00"] == 0:
                    continue

                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])

                # Distance from image center (normalized)
                dist_from_center = np.sqrt(
                    ((cx - center_x) / width) ** 2 +
                    ((cy - center_y) / height) ** 2
                )

                # Simplify contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Generate points string
                points = " ".join(
                    [f"{pt[0][0]:.1f},{pt[0][1]:.1f}" for pt in approx])

                # Calculate importance (area, proximity to center, complexity)
                # importance = area * (1 - dist_from_center) * (1 / (len(approx) + 1))
                importance = 0.5 * area + 0.3 * \
                    (1 - dist_from_center) + 0.2 * len(approx)

                color_features.append(
                    {
                        "points": points,
                        "color": hex_color,
                        "area": area,
                        "importance": importance,
                        "point_count": len(approx),
                        "original_contour": approx,  # Store original contour for adaptive simplification
                    }
                )

            # Sort features by importance within this color
            color_features.sort(key=lambda x: x["importance"], reverse=True)
            hierarchical_features.extend(color_features)

        # Final sorting by overall importance
        hierarchical_features.sort(key=lambda x: x["importance"], reverse=True)

        return hierarchical_features

    def _simplify_polygon(self, points_str, simplification_level):
        """
        Simplify a polygon by reducing coordinate precision or number of points

        Args:
            points_str (str): Space-separated "x,y" coordinates
            simplification_level (int): Level of simplification (0-3)

        Returns:
            str: Simplified points string
        """
        if simplification_level == 0:
            return points_str

        points = points_str.split()

        # Level 1: Round to 1 decimal place
        if simplification_level == 1:
            return " ".join(
                [
                    f"{float(p.split(',')[0]):.1f},{float(p.split(',')[1]):.1f}"
                    for p in points
                ]
            )

        # Level 2: Round to integer
        if simplification_level == 2:
            return " ".join(
                [
                    f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}"
                    for p in points
                ]
            )

        # Level 3: Reduce number of points (keep every other point, but ensure at least 3 points)
        if simplification_level == 3:
            if len(points) <= 4:
                # If 4 or fewer points, just round to integer
                return " ".join(
                    [
                        f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}"
                        for p in points
                    ]
                )
            else:
                # Keep approximately half the points, but maintain at least 3
                step = min(2, len(points) // 3)
                reduced_points = [points[i]
                                  for i in range(0, len(points), step)]
                # Ensure we keep at least 3 points and the last point
                if len(reduced_points) < 3:
                    reduced_points = points[:3]
                if points[-1] not in reduced_points:
                    reduced_points.append(points[-1])
                return " ".join(
                    [
                        f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}"
                        for p in reduced_points
                    ]
                )

        return points_str

    def bitmap_to_svg_layered(
        self,
        image,
        limit=10000,
        resize=True,
        target_size=(384, 384),
        adaptive_fill=True,
        num_colors=None,
    ):
        """
        Convert bitmap to SVG using layered feature extraction with optimized space usage

        Args:
            image: Input image (PIL.Image)
            limit (int): Maximum SVG size
            resize (bool): Whether to resize the image before processing
            target_size (tuple): Target size for resizing (width, height)
            adaptive_fill (bool): Whether to adaptively fill available space
            num_colors (int): Number of colors to quantize, if None uses adaptive selection

        Returns:
            str: SVG representation
        """
        # Adaptive color selection based on image complexity
        if num_colors is None:
            # Simple heuristic: more colors for complex images
            if resize:
                pixel_count = target_size[0] * target_size[1]
            else:
                pixel_count = image.size[0] * image.size[1]

            if pixel_count < 65536:  # 256x256
                num_colors = 8
            elif pixel_count < 262144:  # 512x512
                num_colors = 12
            else:
                num_colors = 16

        # Resize the image if requested
        if resize:
            original_size = image.size
            image = image.resize(target_size, Image.LANCZOS)
        else:
            original_size = image.size

        # Convert to numpy array
        img_np = np.array(image)

        # Get image dimensions
        height, width = img_np.shape[:2]

        # Calculate average background color
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            avg_bg_color = np.mean(img_np, axis=(0, 1)).astype(int)
            bg_hex_color = self._compress_hex_color(
                f"#{avg_bg_color[0]:02x}{avg_bg_color[1]:02x}{avg_bg_color[2]:02x}"
            )
        else:
            bg_hex_color = "#fff"

        # Start building SVG
        # Use original dimensions in viewBox for proper scaling when displayed
        orig_width, orig_height = original_size
        svg_header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{orig_width}" height="{orig_height}" viewBox="0 0 {width} {height}">\n'
        svg_bg = f'<rect width="{width}" height="{height}" fill="{bg_hex_color}"/>\n'
        svg_base = svg_header + svg_bg
        svg_footer = "</svg>"

        # Calculate base size
        base_size = len((svg_base + svg_footer).encode("utf-8"))
        # available_bytes = limit - base_size

        # Extract hierarchical features
        features = self._extract_features_by_scale(
            img_np, num_colors=num_colors)

        # If not using adaptive fill, just add features until we hit the limit
        if not adaptive_fill:
            svg = svg_base
            for feature in features:
                # Try adding the feature
                feature_svg = (
                    f'<polygon points="{feature["points"]}" fill="{feature["color"]}" />\n'
                )

                # Check if adding this feature exceeds size limit
                if len((svg + feature_svg + svg_footer).encode("utf-8")) > limit:
                    break

                # Add the feature
                svg += feature_svg

            # Close SVG
            svg += svg_footer
            return svg

        # For adaptive fill, use binary search to find optimal simplification level

        # First attempt: calculate size of all features at different simplification levels
        feature_sizes = []
        for feature in features:
            feature_sizes.append(
                {
                    "original": len(
                        f'<polygon points="{feature["points"]}" fill="{feature["color"]}" />\n'.encode(
                            "utf-8"
                        )
                    ),
                    "level1": len(
                        f'<polygon points="{self._simplify_polygon(feature["points"], 1)}" fill="{feature["color"]}" />\n'.encode(
                            "utf-8"
                        )
                    ),
                    "level2": len(
                        f'<polygon points="{self._simplify_polygon(feature["points"], 2)}" fill="{feature["color"]}" />\n'.encode(
                            "utf-8"
                        )
                    ),
                    "level3": len(
                        f'<polygon points="{self._simplify_polygon(feature["points"], 3)}" fill="{feature["color"]}" />\n'.encode(
                            "utf-8"
                        )
                    ),
                }
            )

        # Two-pass approach: first add most important features, then fill remaining space
        svg = svg_base
        bytes_used = base_size
        added_features = set()

        # Pass 1: Add most important features at original quality
        for i, feature in enumerate(features):
            feature_svg = (
                f'<polygon points="{feature["points"]}" fill="{feature["color"]}" />\n'
            )
            feature_size = feature_sizes[i]["original"]

            if bytes_used + feature_size <= limit:
                svg += feature_svg
                bytes_used += feature_size
                added_features.add(i)

        # Pass 2: Try to add remaining features with progressive simplification
        for level in range(1, 4):  # Try simplification levels 1-3
            for i, feature in enumerate(features):
                if i in added_features:
                    continue

                feature_size = feature_sizes[i][f"level{level}"]
                if bytes_used + feature_size <= limit:
                    feature_svg = f'<polygon points="{self._simplify_polygon(feature["points"], level)}" fill="{feature["color"]}" />\n'
                    svg += feature_svg
                    bytes_used += feature_size
                    added_features.add(i)

        # Finalize SVG
        svg += svg_footer

        # Double check we didn't exceed limit
        final_size = len(svg.encode("utf-8"))
        if final_size > limit:
            # If we somehow went over, return basic SVG
            return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="{bg_hex_color}"/></svg>'

        return svg

    def convert_all(self, images: list[Image.Image], max_workers=4, limit=10000) -> list[str]:

        svgs = []
        convert_func = partial(self.bitmap_to_svg_layered, limit=limit)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for i, svg in enumerate(executor.map(convert_func, images), 1):
                svgs.append(svg)
                # Log progress every 10 images or at the end
        return svgs

    def process(self, images, limit=20000):
        return self.convert_all(images, limit=limit)
