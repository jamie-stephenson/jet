from torch.optim.lr_scheduler import OneCycleLR, _warn_get_lr_called_within_step
import warnings

class OverflowOneCycleLR(OneCycleLR):
    """OneCycleLR made tolerant to estimated `total_steps`"""
    def get_lr(self):

        _warn_get_lr_called_within_step(self)

        lrs = []
        step_num = self.last_epoch

        # Check for overflow
        if step_num > self.total_steps:
            warnings.warn("The given `total_steps` ({}) has been exceeded, constant learning rate will now be used."
                          .format(self.total_steps))
            final_lrs = [group['lr'] for group in self.optimizer.param_groups]
            if self.cycle_momentum:
                final_momentum = [group['momentum'] if not self.use_beta1 else group['betas'][0] 
                                  for group in self.optimizer.param_groups]
                for group, momentum in zip(self.optimizer.param_groups, final_momentum):
                    if self.use_beta1:
                        group['betas'] = (momentum, *group['betas'][1:])
                    else:
                        group['momentum'] = momentum
            return final_lrs

        for group in self.optimizer.param_groups:
            start_step = 0.0
            for i, phase in enumerate(self._schedule_phases):
                end_step = phase["end_step"]
                if step_num <= end_step or i == len(self._schedule_phases) - 1:
                    pct = (step_num - start_step) / (end_step - start_step)
                    computed_lr = self._anneal_func(
                        group[phase["start_lr"]], group[phase["end_lr"]], pct
                    )
                    if self.cycle_momentum:
                        computed_momentum = self._anneal_func(
                            group[phase["start_momentum"]],
                            group[phase["end_momentum"]],
                            pct,
                        )
                    break
                start_step = phase["end_step"]

            lrs.append(computed_lr)  # type: ignore[possibly-undefined]
            if self.cycle_momentum:
                if self.use_beta1:
                    group["betas"] = (computed_momentum, *group["betas"][1:])  # type: ignore[possibly-undefined]
                else:
                    group[
                        "momentum"
                    ] = computed_momentum  # type: ignore[possibly-undefined]

        return lrs

def get_lr_scheduler(
    optimizer, 
    lr_max, 
    pct_start,
    steps_per_epoch, 
    epochs
):

    scheduler = OverflowOneCycleLR(
        optimizer, 
        lr_max, 
        pct_start=pct_start,
        epochs=epochs, 
        steps_per_epoch=steps_per_epoch
    ) 

    return scheduler