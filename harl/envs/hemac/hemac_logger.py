import numpy as np

from harl.common.base_logger import BaseLogger


class HeMACLogger(BaseLogger):
    def get_task_name(self):
        return self.env_args.get("task", "hemac")

    def init(self, episodes):
        super().init(episodes)
        self.train_episode_metrics = self._empty_episode_metrics()

    def per_step(self, data):
        super().per_step(data)
        dones = data[3]
        infos = data[4]

        for thread_id, done in enumerate(np.all(dones, axis=1)):
            if done:
                self._record_episode_metrics(
                    self.train_episode_metrics,
                    infos[thread_id][0]["hemac_episode"],
                )

    def episode_log(
        self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer
    ):
        super().episode_log(
            actor_train_infos, critic_train_info, actor_buffer, critic_buffer
        )
        self._log_episode_metrics("train", self.train_episode_metrics)

    def eval_init(self):
        super().eval_init()
        self.eval_episode_metrics = self._empty_episode_metrics()

    def eval_thread_done(self, tid):
        super().eval_thread_done(tid)
        self._record_episode_metrics(
            self.eval_episode_metrics,
            self.eval_infos[tid][0]["hemac_episode"],
        )

    def eval_log(self, eval_episode):
        super().eval_log(eval_episode)
        self._log_episode_metrics("eval", self.eval_episode_metrics)

    @staticmethod
    def _empty_episode_metrics():
        return {
            "targets_reached_per_episode": [],
            "target_episode_rate": [],
            "time_to_first_target_success": [],
            "time_to_first_target_capped": [],
            "collision_rate": [],
            "episode_length": [],
            "targets_delivered_per_episode": [],
        }

    def _record_episode_metrics(self, metrics, episode_info):
        metrics["targets_reached_per_episode"].append(
            episode_info["targets_reached"]
        )
        metrics["target_episode_rate"].append(episode_info["episode_had_target"])
        metrics["collision_rate"].append(episode_info["collision_termination"])
        metrics["episode_length"].append(episode_info["episode_length"])
        metrics["targets_delivered_per_episode"].append(
            episode_info["targets_delivered"]
        )

        first_target_step = episode_info["first_target_reached_step"]
        if first_target_step is not None:
            metrics["time_to_first_target_success"].append(first_target_step)
        metrics["time_to_first_target_capped"].append(
            first_target_step
            if first_target_step is not None
            else self.env_args["max_cycles"]
        )

    def _log_episode_metrics(self, prefix, metrics):
        if not metrics["targets_reached_per_episode"]:
            return

        self.log_env(
            {f"{prefix}/{name}": values for name, values in metrics.items()}
        )
        for values in metrics.values():
            values.clear()
