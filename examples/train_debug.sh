python -Xfrozen_modules=off -m debugpy --listen 0.0.0.0:5678 --wait-for-client train.py \
	--load_config "../results/hemac/hemac/mappo/012_more_actions_again_bigger_area/seed-00001-2026-08-25-13-42-38/config.json" \
	--model_dir "../results/hemac/hemac/mappo/012_more_actions_again_bigger_area/seed-00001-2026-08-25-13-42-38/models" \
	--exp_name debug \
	--torch_threads 1 \
	--n_eval_rollout_threads 1 \
	--n_rollout_threads 1 \
	--use_render True
