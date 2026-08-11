python -Xfrozen_modules=off -m debugpy --listen 0.0.0.0:5678 --wait-for-client train.py \
	--exp_name debug \
	--torch_threads 1 \
	--n_eval_rollout_threads 1 \
	--n_rollout_threads 1 \
	--load_config examples/train_config.json
