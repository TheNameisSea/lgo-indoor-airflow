"""EquiJetTrainer — thin subclass of HybridGINOTTrainer (PLAN_EQUIJET_UPGRADE.md §3.4, §6.3).

As with the HBC upgrade, the per-case `set_case` hook the plan anticipated turned
out to be unnecessary: `EquiJetModel` resolves its own case from `pc` inside
`encode_geometry`, so the training loop runs unchanged. What remains here is what
genuinely belongs to training:

  * register vents (and, when composed, wall clouds) for every dataset up front,
    so a missing case fails loudly before training rather than mid-epoch;
  * refuse a hard-BC composition that also has soft wall penalties active;
  * optionally freeze the trunk for the first N epochs so the template fits the
    jets before the correction starts absorbing them (`--freeze_trunk_epochs`).

No existing file is modified.
"""

from pi_ginot.trainer import HybridGINOTTrainer


class EquiJetTrainer(HybridGINOTTrainer):
    def __init__(self, *args, freeze_trunk_epochs=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_trunk_epochs = int(freeze_trunk_epochs)

    def train(self, dataset, val_dataset=None, **kwargs):
        model = self.model

        n = model.register_cases(dataset)
        if val_dataset is not None:
            n += model.register_cases(val_dataset)
        print(f"[equijet] {model.describe()}")

        if model.hbc is not None and kwargs.get("lambda_wall", 0.0):
            raise ValueError(
                "composed EquiJet+HBC enforces no-slip by construction, but "
                f"lambda_wall={kwargs['lambda_wall']} is also set. Use "
                "--lambda_wall 0 so the A/B is unambiguous.")

        # Optional curriculum: template-only warm-up. The trunk is restored before
        # the bulk of training, so the final model is not capacity-limited.
        if self.freeze_trunk_epochs > 0:
            print(f"[equijet] freezing trunk for the first "
                  f"{self.freeze_trunk_epochs} epochs (template-only warm-up)")
            for p in model.trunk.parameters():
                p.requires_grad_(False)
            warm = dict(kwargs)
            warm["epochs"] = self.freeze_trunk_epochs
            super().train(dataset, val_dataset=val_dataset, **warm)
            for p in model.trunk.parameters():
                p.requires_grad_(True)
            kwargs["epochs"] = max(1, kwargs.get("epochs", 2000) - self.freeze_trunk_epochs)

        return super().train(dataset, val_dataset=val_dataset, **kwargs)
