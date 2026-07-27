python -Xfrozen_modules=off  train.py \
--algo mappo \
--env hemac \
--exp_name AA \
--share_param False \
--load_config ./../tuned_configs/hemac/ippo/config.json \
--torch_threads 1 \
--n_eval_rollout_threads 1 \
--n_rollout_threads 1
