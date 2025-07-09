import tempfile
import xml.etree.ElementTree as etree

import pydiffvg
import torch
from tqdm import tqdm
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from typing import Any, Optional, Callable
from utils.image_utils import image_to_tensor

from .base import ISVGOptimizer
from .components.aesthetic import AestheticEvaluatorTorch
from PIL import Image


class DiffVGOptimizer(ISVGOptimizer):
    def __init__(
        self,
        aesthetic_evaluator: Optional[Callable] = None,
        sematic_evaluator: Optional[Callable] = None,
        seed: int = 43,
        device: str = "cuda",
    ):
        self.seed = seed
        self.device = device
        self.aesthetic_evaluator = aesthetic_evaluator
        self.sematic_evaluator = sematic_evaluator

    def _render_svg(self, canvas_width, canvas_height, shapes, shape_groups):
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )

        img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
        alpha = img[:, :, 3:4]
        img = alpha * img[:, :, :3] + (1 - alpha)
        img = img[:, :, :3].unsqueeze(0).permute(0, 3, 1, 2)
        return img

    def _get_scheduler(self, optimizer, cosine_schedule, warmup_steps, n_iter):
        if cosine_schedule:
            return get_cosine_schedule_with_warmup(optimizer, warmup_steps, n_iter)
        return get_constant_schedule_with_warmup(optimizer, warmup_steps)

    def _prepare_optim_params(self, shapes, shape_groups):
        points_vars = []
        initial_points = {}

        for path in shapes:
            if not isinstance(path, pydiffvg.Rect):
                path.points.requires_grad = True
                points_vars.append(path.points)
                initial_points[path.points.data_ptr(
                )] = path.points.data.clone()

        color_vars = []
        initial_colors = {}

        for group in shape_groups:
            group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)
            initial_colors[group.fill_color.data_ptr(
            )] = group.fill_color.data.clone()

        return points_vars, initial_points, color_vars, initial_colors

    def _optimize(
        self,
        svg,
        image,
        n_iter=100,
        point_lr=2.0,
        color_lr=0.05,
        warmup_steps=0,
        cosine_schedule=False,
        optimizer=torch.optim.Adam,
        max_color_deviation=None,
        color_decay=0.0,
        point_decay=0.0,
    ):
        pydiffvg.get_use_gpu()
        loss_history = {
            "total": [],
            "mse": [],
            "aesthetic": []
        }

        target = image_to_tensor([image]).to(self.device)

        root = etree.fromstring(svg)
        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.parse_scene(
            root)

        points_vars, initial_points, color_vars, initial_colors = (
            self._prepare_optim_params(shapes, shape_groups)
        )

        points_optim = optimizer(points_vars, lr=point_lr)
        color_optim = optimizer(color_vars, lr=color_lr)
        points_sched = self._get_scheduler(
            points_optim, cosine_schedule, warmup_steps, n_iter
        )
        color_sched = self._get_scheduler(
            color_optim, cosine_schedule, warmup_steps, n_iter
        )

        best_svg = None

        for t in tqdm(range(n_iter)):
            points_optim.zero_grad()
            color_optim.zero_grad()

            img = self._render_svg(
                canvas_width, canvas_height, shapes, shape_groups)

            aesthetic_loss = -self.aesthetic_evaluator.score(img)
            mse_loss = torch.abs(img - target).mean()
            loss = 0.2 * mse_loss + 0.8 * aesthetic_loss
            loss_history["total"].append(loss.item())
            loss_history["aesthetic"].append(aesthetic_loss.item())
            loss_history["mse"].append(mse_loss.item())
            loss.backward()

            points_optim.step()
            color_optim.step()
            points_sched.step()
            color_sched.step()

            for i, group in enumerate(shape_groups):
                initial_color = initial_colors[group.fill_color.data_ptr()]
                if color_decay > 0:
                    group.fill_color.data = (
                        color_decay * group.fill_color.data
                        + (1 - color_decay) * initial_color
                    )
                group.fill_color.data.clamp_(0.0, 1.0)
                if i == 0:
                    group.fill_color.data[-1] = 1.0
                if max_color_deviation is not None:
                    min_color = torch.clamp(
                        initial_color - max_color_deviation, 0.0, 1.0
                    )
                    max_color = torch.clamp(
                        initial_color + max_color_deviation, 0.0, 1.0
                    )
                    group.fill_color.data.clamp_(min_color, max_color)

            for point in points_vars:
                initial_point = initial_points[point.data_ptr()]
                if point_decay > 0:
                    point.data = (
                        point_decay * point.data +
                        (1 - point_decay) * initial_point
                    )
                point.data.clamp_(0.0, canvas_height)

        if best_svg is not None:
            return best_svg
        with tempfile.NamedTemporaryFile("r+", delete=False, suffix=".svg") as tmpfile:
            pydiffvg.save_svg(
                tmpfile.name, canvas_width, canvas_height, shapes, shape_groups
            )
            tmpfile.seek(0)
            return tmpfile.read(), loss_history

    def process(self, svg_code: str):
        pass


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    aesthetic_evaluator = AestheticEvaluatorTorch()
    diffvg_optimizer = DiffVGOptimizer(aesthetic_evaluator)
    with open("/home/anhndt/pysvgenius/data/test/test_svg.svg", "r") as f:
        svg_content = f.read()
    image = Image.open("/home/anhndt/pysvgenius/data/test/Screenshot 2025-05-29 175550.png").convert("RGB")
    resized_image = image.resize((384, 384), Image.Resampling.LANCZOS)
    optimized_svg , loss_history= diffvg_optimizer._optimize(
        svg_content, resized_image, n_iter=200)

    with open("svg_01.svg", "w") as f:
        f.write(optimized_svg)

    # Vẽ biểu đồ
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(loss_history["total"], label="Total Loss")
    plt.plot(loss_history["mse"], label="MSE Loss")
    plt.plot(loss_history["aesthetic"], label="Aesthetic Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("SVG Optimization Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")   # Lưu ra file nếu cần
    plt.show()      