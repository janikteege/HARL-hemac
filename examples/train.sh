experiment_name="${1:?Usage: $0 EXPERIMENT [options]}"
shift
python train.py \
	--exp_name "${experiment_name}" \
	--load_config "../experiments/${experiment_name}.json" \
	"$@"
