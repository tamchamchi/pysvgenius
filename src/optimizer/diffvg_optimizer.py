import tempfile
from pathlib import Path

import numpy as np
import pydiffvg
import torch
from PIL import Image
from transformers import AutoModel
from .base import ISVGOptimizer
from torchvision import transforms
from tqdm import trange

import time
from .components.image_processor_torch import ImageProcessorTorch
from .components.aesthetic import AestheticEvaluatorTorch

import torch.nn.functional as F
import os

from ..utils.logger import get_library_logger
from typing import Optional
import logging


class DiffVGOptimizer(ISVGOptimizer):
    def __init__(self, logger: Optional[logging.Logger] = None, device=torch.device("cuda:0"), seed: int = 43):
        if logger is not None:
            self.logger = logger
        else:
            self.logger = get_library_logger(
                f"{__name__}.{self.__class__.__name__}")

        self.device = device
        self.seed = seed

    def _load_config(self, cfg):
        pass

    def _load_svg_and_prepare_params(
        self, svg_path: Path
    ) -> tuple[int, int, list, list, dict] | None:
        """Load an SVG file and collect parameters for optimization"""
        if not svg_path.exists():
            self.logger.warning(f"SVG file not found: {svg_path}")
            return None
        try:
            canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
                str(svg_path)
            )

            # Resize the canvas to 384x384
            assert canvas_width == canvas_height, "Image must be square"
            ratio = 384 / canvas_width
            canvas_width = 384
            canvas_height = 384
            for shape in shapes:
                if hasattr(shape, "points") and shape.points is not None:
                    shape.points = shape.points * ratio
                if hasattr(shape, "center") and shape.center is not None:
                    shape.center = shape.center * ratio
                if hasattr(shape, "radius") and shape.radius is not None:
                    shape.radius = shape.radius * ratio
                if hasattr(shape, "p_min") and shape.p_min is not None:
                    shape.p_min = shape.p_min * ratio
                if hasattr(shape, "p_max") and shape.p_max is not None:
                    shape.p_max = shape.p_max * ratio
                if hasattr(shape, "stroke_width") and shape.stroke_width is not None:
                    shape.stroke_width = shape.stroke_width * ratio

            shape_groups = [g for g in shape_groups if g.shape_ids.numel() > 0]
            self.logger.info(
                f"Loaded SVG file. Canvas: {canvas_width}x{canvas_height}")

            points_vars = []
            stroke_width_vars = []
            color_vars = []

            # Extract parameters to be optimized from all but the background and overlay elements
            for shape in shapes[1:-2]:
                if hasattr(shape, "points") and shape.points is not None:
                    shape.points.requires_grad = True
                    points_vars.append(shape.points)
                if hasattr(shape, "center") and shape.center is not None:  # Circle
                    shape.center.requires_grad = True
                    points_vars.append(shape.center)
                if hasattr(shape, "radius") and shape.radius is not None:  # Circle
                    shape.radius.requires_grad = True
                    points_vars.append(shape.radius)
                if hasattr(shape, "p_min") and shape.p_min is not None:  # Rect
                    shape.p_min.requires_grad = True
                    points_vars.append(shape.p_min)
                if hasattr(shape, "p_max") and shape.p_max is not None:  # Rect
                    shape.p_max.requires_grad = True
                    points_vars.append(shape.p_max)
                if hasattr(shape, "stroke_width") and shape.stroke_width is not None:
                    shape.stroke_width.requires_grad = True
                    stroke_width_vars.append(shape.stroke_width)

            for group in shape_groups[:-2]:
                if hasattr(group, "fill_color") and group.fill_color is not None:
                    group.fill_color.requires_grad = True
                    color_vars.append(group.fill_color)
                if hasattr(group, "stroke_color") and group.stroke_color is not None:
                    group.stroke_color.requires_grad = True
                    color_vars.append(group.stroke_color)

            self.logger.info(
                f"Parameters to optimize: points={len(points_vars)}, stroke_width={len(stroke_width_vars)}, color={len(color_vars)}"
            )
            if not points_vars and not stroke_width_vars and not color_vars:
                self.logger.warning("No optimizable parameters found.")
                return None

            params = {
                "points_vars": points_vars,
                "stroke_width_vars": stroke_width_vars,
                "color_vars": color_vars,
            }
            return canvas_width, canvas_height, shapes, shape_groups, params

        except Exception as e:
            self.logger.error(f"Error occurred while loading SVG file: {e}")
            return None

    def _setup_optimizer(
        self, args, params: dict
    ) -> tuple[torch.optim.Optimizer, list[dict]] | None:
        """Set up the optimizer and parameter groups"""
        coords_initial_lr = args.lr_points
        coords_final_lr = args.lr_points / 10
        color_initial_lr = args.lr_color
        color_final_lr = args.lr_color / 10
        # Learning rate for stroke width (can be configurable via args)
        stroke_initial_lr = 0
        stroke_final_lr = 0

        param_groups = []
        if params["points_vars"]:
            param_groups.append(
                {
                    "params": params["points_vars"],
                    "lr": coords_initial_lr,
                    "name": "coords",
                    "initial_lr": coords_initial_lr,
                    "final_lr": coords_final_lr,
                }
            )
        if params["stroke_width_vars"]:
            param_groups.append(
                {
                    "params": params["stroke_width_vars"],
                    "lr": stroke_initial_lr,
                    "name": "stroke",
                    "initial_lr": stroke_initial_lr,
                    "final_lr": stroke_final_lr,
                }
            )
        if params["color_vars"]:
            param_groups.append(
                {
                    "params": params["color_vars"],
                    "lr": color_initial_lr,
                    "name": "color",
                    "initial_lr": color_initial_lr,
                    "final_lr": color_final_lr,
                }
            )

        if not param_groups:
            self.logger.error("No valid parameter groups found.")
            return None

        optimizer = torch.optim.Adam(param_groups)
        return optimizer, param_groups

    def _calculate_target_image_embedding_with_siglip(
        self, image: Image, model: AutoModel, device: str, dtype: torch.dtype
    ) -> torch.Tensor | None:
        """Load the target image and compute SIGLIP image embedding"""

        image_size = model.config.vision_config.image_size
        target_img_pil = image.convert("RGB").resize((image_size, image_size))

        # Using processor is more robust
        # inputs = processor(images=target_img_pil, return_tensors="pt").to(device=device, dtype=dtype)
        # pixel_values = inputs["pixel_values"]

        target_img_np = np.array(target_img_pil)
        target_img_tensor = torch.from_numpy(target_img_np)

        # Convert HWC -> BCHW
        target_img_tensor = target_img_tensor.permute(2, 0, 1)
        target_img_tensor = target_img_tensor.unsqueeze(0)

        # Convert to float and normalize to [-1.0, 1.0]
        target_img_tensor = target_img_np.float() / 255.0
        target_img_torch = target_img_np * 2.0 - 1.0

        # Fast version
        pixel_values = target_img_torch.to(device=device, dtype=dtype)

        with torch.no_grad():
            target_embedding = model.get_image_features(
                pixel_values=pixel_values)
        target_embedding.requires_grad_(False)  # Gradients not needed
        self.logger.info(
            f"Computed encoding for target image. Shape: {target_embedding.shape}"
        )
        return target_embedding

    def _render(
        self, canvas_width, canvas_height, shapes, shape_groups, seed=0
    ) -> torch.Tensor:
        """Render the scene using pydiffvg"""

        _render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )
        img = _render(canvas_width, canvas_height,
                      2, 2, seed, None, *scene_args)
        return img

    def _get_lr(
        self, initial_lr, final_lr, iteration, warmup_iterations, total_iterations
    ):
        # Cosine Annealing with Warmup learning rate scheduler
        if iteration < warmup_iterations:
            return initial_lr * (iteration + 1) / warmup_iterations
        progress = (iteration - warmup_iterations) / (
            total_iterations - warmup_iterations
        )
        return final_lr + (initial_lr - final_lr) * 0.5 * (1 + np.cos(np.pi * progress))

    def run(
        self,
        args,
        aesthetic_evaluator_torch,
        similarity_mode,
        siglip_model=None,
        target_image_embedding=None,
        device=torch.device("cuda:0"),
        dtype=torch.float16,
        image=None,
    ):
        # Record start time and initialize best loss
        start_time = time.time()
        best_loss = float("inf")

        best_iteration = 0
        warmup_iteration = args.warmup_iter
        batch_size = args.batch_size

        # Convert the target PIL image to tensor and move to device
        image = image.resize((self.canvas_width, self.canvas_height))
        to_tensor = transforms.ToTensor()
        tensor_image = to_tensor(image)
        tensor_image = tensor_image.to(device)

        # Create two image processors: one for processing renderings, one for augmentations
        seed = int(time.time() * 1000) % (2**32 - 1)
        image_processor_torch = ImageProcessorTorch(seed=seed).to(device)
        image_processor_torch_ref = ImageProcessorTorch(seed=seed).to(device)

        # Main optimization loop
        for iter in trange(args.iterations):
            # --- Learning rate scheduling ---
            for param_group in self.optimizer.param_groups:
                group_init_lr = param_group["initial_lr"]
                group_final_lr = param_group["final_lr"]
                current_lr = self._get_lr(
                    group_init_lr,
                    group_final_lr,
                    iter,
                    warmup_iteration,
                    args.iterations,
                )
                param_group["lr"] = current_lr

            self.optimizer.zero_grad()

            # --- Render current SVG scene into an image tensor (1, C, H, W) ---
            img_render_single = (
                self._render(
                    self.canvas_width,
                    self.canvas_height,
                    self.shapes,
                    self.shape_groups,
                    seed,
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device, dtype=dtype)
            )
            # --- Batch duplication for processing ---
            img_render_batch = img_render_single.repeat(
                batch_size, 1, 1, 1
            )  # (B, C, H, W) of svg

            # --- Apply JPEG compression simulation if beyond warm-up stage ---
            if iter < args.jpeg_iter:
                processed_image_render_batch = image_processor_torch.apply(
                    img_render_batch, skip_jpeg_compression=True
                )
            else:
                processed_image_render_batch = image_processor_torch.apply(
                    img_render_batch
                )

            # --- Prepare variables for loss computation ---
            similarity_loss_raw = torch.tensor(
                np.inf, device=device, dtype=dtype)
            similarity_loss = torch.tensor(0.0, device=device, dtype=dtype)

            # --- Prepare target image batch and augment with random crop/resize ---
            tensor_image_batch = tensor_image.repeat(batch_size, 1, 1, 1)
            tensor_image_cropped_batch = (
                image_processor_torch_ref.apply_random_crop_resize(
                    tensor_image_batch)
            )

            mse_loss = (
                (processed_image_render_batch - tensor_image_cropped_batch)
                .pow(2)
                .mean()
            )

            # --- Compute similarity loss using SIGLIP model ---
            try:
                # (B, C, H, W), [0,1] -> (B, C, H, W), [-1, 1]
                # The SIGLIP processor performs normalization internally, so you can either pass [0,1] or manually match the model's expected input.
                # Here we follow exp028 and manually normalize to [-1, 1].
                render_pixel_values_batch = (
                    processed_image_render_batch * 2.0) - 1.0
                render_pixel_values_batch = render_pixel_values_batch.to(
                    dtype=siglip_model.dtype
                )  # Match the dtype of the SIGLIP model

                # Resize to the input size expected by siglip_model
                render_pixel_values_batch = F.interpolate(
                    render_pixel_values_batch,
                    size=siglip_model.config.vision_config.image_size,
                    mode="bilinear",
                    align_corners=False,
                )
                # Extract image embeddings and compute cosine similarity with target
                rendered_image_embedding_batch = siglip_model.get_image_features(
                    pixel_values=render_pixel_values_batch
                )  # (B, embed_dim)

                cosine_similarity_batch = torch.nn.functional.cosine_similarity(
                    target_image_embedding, rendered_image_embedding_batch, dim=-1
                )  # (B,)
                similarity_loss_raw = (
                    -cosine_similarity_batch.mean()
                )  # Negative sign for loss
                # Normalization may not be necessary for SIGLIP, needs further consideration
                similarity_loss = similarity_loss_raw

            except Exception as e:
                self.logger.error(
                    f"Exception occurred during SIGLIP similarity calculation (iter {iter}): {e}"
                )
                break

            # --- Compute aesthetic score and loss ---
            try:
                # Pass processed_img_render_batch to the aesthetic_evaluator_torch
                # The resolution can be different from PaliGemma (AestheticEvaluatorTorch handles resizing internally)
                # processed_img_render_batch is in shape BCHW, range [0,1], with the main dtype
                aesthetic_score_batch = aesthetic_evaluator_torch.score(
                    processed_image_render_batch.to(
                        device, dtype=torch.float16)
                )  # (B,)
                aesthetic_loss = -aesthetic_score_batch.mean()
            except Exception as e:
                self.logger.error(
                    f"Failed to compute aesthetic score (iter {iter}): {e}"
                )
                break

            # --- Total loss: depends on current iteration ---
            if iter >= args.aesthetic_iter:
                if similarity_mode == "siglip":
                    loss = (
                        args.w_aesthetic * aesthetic_loss
                        + args.w_siglip * similarity_loss
                        + args.w_mse * mse_loss
                    )
                else:  # Should not happen
                    loss = args.w_aesthetic * aesthetic_loss + args.w_mse * mse_loss
            else:  # During aesthetic_iter, use only aesthetic loss
                loss = args.w_aesthetic * aesthetic_loss

            # --- Abort if loss becomes NaN ---
            if torch.isnan(loss):
                self.logger.error(
                    f"Total loss became NaN (iter {iter}). Stopping optimization."
                )
                break

            # --- Backpropagation and custom gradient filtering ---
            try:
                loss.backward()
                with torch.no_grad():
                    for group in self.shape_groups:
                        # Prevent gradients from affecting alpha channels
                        if (
                            hasattr(group, "fill_color")
                            and group.fill_color is not None
                            and group.fill_color.grad is not None
                        ):
                            # Ignore alpha channel in gradient
                            group.fill_color.grad[3] = 0.0
                        if (
                            hasattr(group, "stroke_color")
                            and group.stroke_color is not None
                            and group.stroke_color.grad is not None
                        ):
                            # Ignore alpha channel in gradient
                            group.stroke_color.grad[3] = 0.0
            except Exception as e:
                self.logger.error(
                    f"Exception occurred during gradient computation (iter {iter}): {e}"
                )
                break

            # --- Gradient clipping and optimizer step ---
            torch.nn.utils.clip_grad_norm_(
                [
                    p
                    for pg in self.param_groups
                    for p in pg["params"]
                    if p.grad is not None
                ],
                max_norm=args.grad_clip_norm,
            )
            self.optimizer.step()

            # --- Post-update clamping to keep parameters within bounds ---
            with torch.no_grad():
                for group in self.shape_groups:
                    if (
                        hasattr(group, "fill_color")
                        and group.fill_color is not None
                        and group.fill_color.requires_grad
                    ):
                        group.fill_color.data.clamp_(0.0, 1.0)
                    if (
                        hasattr(group, "stroke_color")
                        and group.stroke_color is not None
                        and group.stroke_color.requires_grad
                    ):
                        group.stroke_color.data.clamp_(0.0, 1.0)
                for shape in self.shapes:
                    if (
                        hasattr(shape, "stroke_width")
                        and shape.stroke_width is not None
                        and shape.stroke_width.requires_grad
                    ):
                        shape.stroke_width.data.relu_()
                    if (
                        hasattr(shape, "radius")
                        and shape.radius is not None
                        and shape.radius.requires_grad
                    ):
                        shape.radius.data.clamp_(min=1.0)

            current_loss = loss.item()

            # If after aesthetic_iter and loss is improved, update best result
            if iter >= args.aesthetic_iter and current_loss < best_loss:
                best_loss = current_loss
                best_iteration = iter

            # Logging
            if (iter + 1) % args.log_interval == 0 or iter == 0:
                with torch.no_grad():
                    current_aesthetic_score_torch_avg = (
                        aesthetic_score_batch.mean().item()
                    )
                    # PaliGemma loss is already scalar
                    current_similarity_loss_raw_avg = similarity_loss_raw.item()
                    current_similarity_loss_avg = similarity_loss.item()

                elapsed_time = time.time() - start_time
                lrs = {pg["name"]: pg["lr"]
                       for pg in self.optimizer.param_groups}
                lr_str = "/".join([f"{lr:.6f}" for lr in lrs.values()])
                self.logger.info(
                    f"Iter [{(iter + 1):>4}/{args.iterations}], "
                    f"LRs: {lr_str}, "
                    f"AesScore(T Avg): {current_aesthetic_score_torch_avg:.6f}, "
                    f"{similarity_mode.upper()}Loss(Raw Avg): {current_similarity_loss_raw_avg:.6f}, "
                    f"{similarity_mode.upper()}Loss(Norm/Direct Avg): {current_similarity_loss_avg:.6f}, "
                    f"Loss(A/Sim/Mse/T Avg): {aesthetic_loss.item():.4f}/{similarity_loss.item():.4f}/{mse_loss.item():.4f}/{loss.item():.6f}, "
                    f"Best Loss: {best_loss:.6f} (iter {best_iteration}), "
                    f"Time: {elapsed_time:.2f}s"
                )

        self.logger.success(
            f"Optimization completed! Best Loss: {best_loss:.6f} at iteration {best_iteration}"
        )
        return self.shapes, self.shape_groups, -aesthetic_loss.item()

    def _load_aesthetic_evaluator(self):
        self.logger.debug("Loading Aesthetic Evaluator ...")
        try:
            aesthetic_evaluator_torch = AestheticEvaluatorTorch()
            self.logger.success("AestheticEvaluatorTorch initialized.")
            return aesthetic_evaluator_torch
        except Exception as e:
            self.logger.error(
                f"Failed to initialize AestheticEvaluatorTorch: {e}")
            raise

    def _load_image_processor_torch(self):
        self.logger.debug("Loading Image Processor Torch ...")
        try:
            image_processor_torch = ImageProcessorTorch(
                seed=self.seed).to(self.device)
            self.logger.success(
                f"ImageProcessorTorch initialized on device: {self.device}")
            return image_processor_torch
        except Exception as e:
            self.logger.error(
                f"Failed to initialize ImageProcessorTorch: {e}")
            raise

    def build(self, similarity_model: str = "siglip"):
        self.logger.debug("Starting DiffVG optimizer build process")
        self.aesthetic_evaluator_torch = self._load_aesthetic_evaluator()
        self.image_processor_torch = self._load_image_processor_torch()
        self.image_processor_torch_ref = self._load_image_processor_torch()
        
    def process(self, svg: str, image: Image.Image, args):
        # Convert svg to svg_file_path
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_file_path = os.path.join(tmp_dir.name, "tmp.svg")
        with open(tmp_file_path, "w") as f:
            f.write(svg)
        self.logger.debug(f"Temporary SVG file created: {tmp_file_path}")

        # Prepare params to optimization
        result = self._load_svg_and_prepare_params(Path(tmp_file_path))
        if result is None:
            self.logger.error("Failed to load SVG and prepare parameters")
            return

        (
            self.canvas_width,
            self.canvas_height,
            self.shapes,
            self.shape_groups,
            params,
        ) = result

        # Set up optimizer
        optimizer_result = self._setup_optimizer(args, params)
        if optimizer_result is None:
            self.logger.error("Failed to setup optimizer")
            return

        self.optimizer, self.param_groups = optimizer_result
        self.logger.success("DiffVG optimizer build completed successfully")
