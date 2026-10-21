cd -- "$(dirname "$0")" || exit 1
experiment_name="${1:?Usage: $0 EXPERIMENT [options]}"
shift

python -Xfrozen_modules=off -m debugpy --listen 0.0.0.0:5678 --wait-for-client train.py \
	--env hemac \
	--algo happo \
	--exp_name "${experiment_name}" \
	--torch_threads 1 \
	--n_eval_rollout_threads 1 \
	--n_rollout_threads 1 \
	--load_config "../experiments/${experiment_name}.json" \
	"$@"
