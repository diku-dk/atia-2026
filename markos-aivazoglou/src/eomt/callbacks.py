"""Lightning callbacks for the EoMT setup (our own code, referenced by class path from ``configs/seg.yaml``)."""

from lightning.pytorch import Callback


class AttnMaskAnnealing(Callback):
    """Set the module's masked-attention annealing steps from fractions of the run's total steps.

    Upstream configs give ``attn_mask_annealing_{start,end}_steps`` as absolute global steps of a
    COCO schedule (~7.4k steps per epoch), which on CropAndWeed's ~5.4k training images would never
    finish annealing. ``start``/``end`` hold one fraction per annealed block instead, converted with
    Lightning's own ``estimated_stepping_batches`` (so epochs, batch size, GPU count and
    ``limit_train_batches`` are all accounted for) before training starts. Masked attention is thus
    fully annealed (all ``attn_mask_prob_*`` 0) by the end of every run.
    """

    def __init__(self, start: list[float], end: list[float]):
        self.start = start
        self.end = end

    def on_fit_start(self, trainer, pl_module) -> None:
        total = trainer.estimated_stepping_batches
        pl_module.attn_mask_annealing_start_steps = [round(f * total) for f in self.start]
        pl_module.attn_mask_annealing_end_steps = [round(f * total) for f in self.end]
        print(
            f"Attention-mask annealing over {total} steps: start {pl_module.attn_mask_annealing_start_steps}, "
            f"end {pl_module.attn_mask_annealing_end_steps}"
        )
