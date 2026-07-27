#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="./../tuned_configs/hemac/ippo/config.json"

for hs in 128 256; do
	  exp="extended_state_concat_hs${hs}"

	  python -Xfrozen_modules=off train.py \
	    --env hemac \
	    --algo mappo \
	    --load_config "${BASE_CONFIG}" \
	    --exp_name "${exp}" \
	    --hidden_sizes "[${hs}","${hs}]"
done
