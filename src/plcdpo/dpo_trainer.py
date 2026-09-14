from .base_trainer import OrderedDPOTrainer
from .losses import dpo_loss


class DPOTrainer(OrderedDPOTrainer):
    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        output = self.concatenated_forward(model, batch)
        ref_chosen, ref_rejected = self._resolve_ref_logps(batch)
        loss, chosen, rejected = dpo_loss(
            output["chosen_logps"],
            output["rejected_logps"],
            ref_chosen,
            ref_rejected,
            float(self.args.beta),
        )
        prefix = "eval_" if train_eval == "eval" else ""

        def mean(value):
            return self.accelerator.gather_for_metrics(value.detach().float()).mean().item()

        metrics = {
            f"{prefix}rewards/chosen": mean(chosen),
            f"{prefix}rewards/rejected": mean(rejected),
            f"{prefix}rewards/margins": mean(chosen - rejected),
            f"{prefix}rewards/accuracies": mean((chosen > rejected).float()),
        }
        return loss.mean(), metrics
