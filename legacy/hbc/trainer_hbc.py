"""HBCTrainer — thin subclass of HybridGINOTTrainer for the hard-BC model.

The plan anticipated having to copy the per-case loop body in order to inject a
`set_case(...)` hook. That turned out to be unnecessary: `HardBCModel` resolves
its own case from `pc` inside `encode_geometry`, so the entire training loop works
unchanged. What is left for this subclass is the part that genuinely belongs to
training rather than to the model:

  * register the datasets' FULL boundary pools before training (otherwise the
    wrapper silently falls back to the subsampled clouds inside `pc`);
  * refuse configurations where a hard constraint and its soft penalty are BOTH
    active, which would make an A/B comparison meaningless.

No existing file is modified.
"""

from pi_ginot.trainer import HybridGINOTTrainer


class HBCTrainer(HybridGINOTTrainer):
    def train(self, dataset, val_dataset=None, **kwargs):
        model = self.model
        stage = getattr(model, "stage", 0)

        # Full boundary pools beat the subsampled clouds carried in `pc`.
        if hasattr(model, "register_cases"):
            n = model.register_cases(dataset)
            n += model.register_cases(val_dataset) if val_dataset is not None else 0
            print(f"[hbc] registered {n} case(s) from full boundary pools")
        print(f"[hbc] {model.describe()}")

        # A hard constraint plus its soft penalty is a silent config error: the
        # penalty is trivially ~0 under the ansatz, so it neither helps nor is
        # visibly inactive. Fail loudly instead.
        if stage >= 1 and kwargs.get("lambda_wall", 0.0):
            raise ValueError(
                f"stage {stage} enforces no-slip walls by construction, but "
                f"lambda_wall={kwargs['lambda_wall']} is also set. Use "
                f"--lambda_wall 0 so the A/B is unambiguous.")
        if stage >= 2 and kwargs.get("lambda_inlet", 0.0):
            raise ValueError(
                f"stage 2 enforces inlet velocities by construction, but "
                f"lambda_inlet={kwargs['lambda_inlet']} is also set. Use "
                f"--lambda_inlet 0.")

        return super().train(dataset, val_dataset=val_dataset, **kwargs)
