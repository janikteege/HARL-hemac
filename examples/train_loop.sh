#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="./../tuned_configs/hemac/ippo/config.json"

for hs in 128 256; do
	for lr in 1e-4 3e-4; do
		for gamma in 0.95 0.99; do
			for ent in 0.01 0.05; do
				  exp="extended_state_hs${hs}_lr${lr}_g${gamma}_ent${ent}"

				  python -Xfrozen_modules=off train.py \
				    --env hemac \
				    --algo mappo \
				    --load_config "${BASE_CONFIG}" \
				    --exp_name "${exp}" \
				    --hidden_sizes "[${hs}","${hs}]" \
				    --lr "${lr}" \
				    --critic_lr "${lr}" \
				    --gamma "${gamma}" \
				    --entropy_coef "${ent}"
			done
		done
	done
done
