import torch
import torch.nn as nn
from typing import Dict, Tuple, List
import numpy as np
from monai.metrics import DiceMetric
from monai.transforms import Compose
from monai.data import decollate_batch
from monai.transforms import Activations
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference


################################################################################
class SlidingWindowInference:
    """Efficient sliding window inference for volumetric segmentation.
    
    Uses MONAI's optimized sliding window implementation with memory-efficient
    batch processing and overlap handling.
    """
    
    def __init__(self, roi: Tuple[int, int, int], sw_batch_size: int, depth_chunks: int = 32) -> None:
        """Initialize sliding window inference.
        
        Args:
            roi: Region of interest size (D, H, W)
            sw_batch_size: Batch size for sliding window patches
        """
        self.dice_metric = DiceMetric(
            include_background=True, 
            reduction="mean_batch", 
            get_not_nans=False,
            ignore_empty=True
        )
        self.to_onehot = Compose(
            [
                AsDiscrete(argmax=True, to_onehot=26),
            ]
        )
        self.to_label = Compose(
            [
                AsDiscrete(argmax=True),
            ]
        )
        self.sw_batch_size = sw_batch_size
        self.roi = roi
        self.depth_chunks = depth_chunks

    def __call__(
        self, 
        val_inputs: torch.Tensor, 
        val_labels: torch.Tensor, 
        model: nn.Module
    ) -> float:
        """Compute Dice metric using sliding window inference.
        
        Args:
            val_inputs: Input volume (B, C, D, H, W)
            val_labels: Ground truth labels (B, C, D, H, W)
            model: Segmentation model
            
        Returns:
            Average Dice score across all classes (percentage)
        """
        
        # Perform sliding window inference
        logits = self.forward(val_inputs, model)
        return self.calc_dice_metric(logits, val_labels.cpu())

    def forward(
        self, 
        val_inputs: torch.Tensor, 
        model: nn.Module
    ) -> torch.Tensor:
        with torch.inference_mode():  # More efficient than no_grad for inference
            logits = sliding_window_inference(
                inputs=val_inputs,
                roi_size=self.roi,
                sw_batch_size=self.sw_batch_size,
                predictor=model,
                overlap=0.167,
                sw_device="cuda",
                device="cpu"
            )
        return logits

    def calc_dice_metric(
        self,
        logits: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> float:
        # Decollate and post-process predictions
        val_labels_list = decollate_batch(val_labels)
        val_outputs_list = decollate_batch(logits)
        val_output_convert = [
            self.to_onehot(val_pred_tensor) for val_pred_tensor in val_outputs_list
        ]

        # reset buffer
        self.dice_metric.reset()
        
        # Compute Dice metric
        self.dice_metric(y_pred=val_output_convert, y=val_labels_list)
        
        # Aggregate results - compute accuracy per channel
        acc = self.dice_metric.aggregate().cpu().numpy()
        avg_acc = float(acc.mean())  # Explicit conversion for clarity
        
        # To access individual metric:
        # TC acc: acc[0]
        # WT acc: acc[1]
        # ET acc: acc[2]
        return avg_acc * 100.0

    def calc_loss_dice_metric(
        self,
        logits: torch.Tensor,
        val_labels: torch.Tensor,
        criterion: nn.Module | None,
        device
    ) -> float:
        """Compute loss and dice metric using sliding window inference.

        Args:
            logits: Logits from the model (B, C, D, H, W)
            val_labels: Ground truth labels (B, C, D, H, W)
            criterion: Loss function

        Returns:
            Tuple of (loss, dice_score)
        """
        losses = []
        dices = []

        for logit, label in zip(decollate_batch(logits), decollate_batch(val_labels)):
            # process in chunks in gpu
            channels, D, H, W = logit.shape
            pred_cpu = torch.zeros((channels, D, H, W), dtype=torch.float32)
            for start_d in range(0, D, self.depth_chunks):
                end_d = min(start_d + self.depth_chunks, D)
                logit_chunk_gpu = logit[:, start_d:end_d, :, :].to(device)
                label_chunk_gpu = label[:, start_d:end_d, :, :].to(device)

                pred_chunk_gpu = self.to_onehot(logit_chunk_gpu)
                pred_cpu[:, start_d:end_d, :, :] = pred_chunk_gpu.cpu()

                if criterion is not None:
                    loss_chunk = criterion(logit_chunk_gpu.unsqueeze(0), label_chunk_gpu.unsqueeze(0))
                    losses.append(loss_chunk.detach().item())

            # compute dice of the full volume excluding unpresent classes
            self.dice_metric.reset()
            self.dice_metric(y_pred=[pred_cpu], y=[label])
            raw_dice = self.dice_metric.aggregate().cpu().numpy()
            present_classes = torch.unique(label)
            dices.append(float(raw_dice[present_classes].mean()))

        # compute loss
        if criterion is not None:
            loss = sum(losses) / len(losses)
        else:
            loss = None

        # compute dice
        avg_acc = sum(dices) / len(dices) * 100.0

        return loss, avg_acc


def build_metric_fn(metric_type: str, metric_arg: Dict) -> SlidingWindowInference:
    """Factory function to build metric computation modules.
    
    Args:
        metric_type: Type of metric ('sliding_window_inference')
        metric_arg: Dictionary containing metric configuration
        
    Returns:
        Instantiated metric module
        
    Raises:
        ValueError: If metric_type is not supported
    """
    if metric_type == "sliding_window_inference":
        return SlidingWindowInference(
            roi=metric_arg["roi"],
            sw_batch_size=metric_arg["sw_batch_size"],
            depth_chunks=metric_arg["depth_chunks"],
        )
    else:
        raise ValueError(
            f"Unsupported metric type: {metric_type}. "
            "Supported types: ['sliding_window_inference']"
        )
