from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class OptimalTransportHook(Hook):

    def __init__(self, interval = 2000, start_emb = 2, is_onestage=True):
        self.step = 0
        self.epoch_record = 0
        self.interval = interval
        self.start_emb = start_emb
        self.end_estimating = 8
        self.is_onestage = is_onestage

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass


    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        self.epoch_record += 1
        if self.epoch_record == self.start_emb:
            runner.model.module.bbox_head.enable_emd_training()
        pass

    def before_iter(self, runner):
        self.step += 1
        if self.step % self.interval == 0 and self.epoch_record < self.end_estimating:
            self.step = 0
            if self.is_onestage:
                runner.model.module.bbox_head.update_ot()
            else:
                runner.model.module.roi_head.bbox_head.update_ot()

    def after_iter(self, runner):
        pass