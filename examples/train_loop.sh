for recn in 1 2; do
	python -Xfrozen_modules=off train.py \
		--env hemac \
		--algo mappo \
		--load_config "./../tuned_configs/hemac/ippo/config.json" \
		--exp_name "new_state_recn${recn}" \
		--recurrent_n $recn
done
