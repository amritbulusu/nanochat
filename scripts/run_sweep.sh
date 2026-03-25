#!/bin/bash
set -e

# Setup
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv && uv sync --extra gpu
source .venv/bin/activate
apt-get update && apt-get install -y python3-dev
python -m nanochat.dataset -n 2
python -m scripts.tok_train

# Launch 8 shards on 8 GPUs (8 LR values, 1 per GPU)
for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i python -m scripts.mup_hp_sweep \
        --sweep-2d \
        --widths 256 \
        --lr-values 0.003,0.005,0.007,0.01,0.015,0.02,0.03,0.05\
        --wd-values 0.0,0.1,0.2,0.3,0.4,0.6 \
        --batch-size 16 \
        --grad-accum-steps 16 \
        --shard $i/8 \
        --save-dir /tmp/sweep_shard_$i \
        --no-plot &
done
echo "All shards launched"
wait
echo "All shards done"
