from harl.common.base_logger import BaseLogger


class HeMACLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args.get("task", "hemac")
