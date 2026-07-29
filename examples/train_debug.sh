python -Xfrozen_modules=off examples/train.py \
--algo mappo \
--env hemac \
--exp_name debug \
--share_param False \
--torch_threads 1 \
--n_eval_rollout_threads 1 \
--n_rollout_threads 1
