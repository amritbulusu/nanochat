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

# Launch 5 shards on 5 GPUs (5 LR values, 1 per GPU)
for i in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$i python -m scripts.mup_hp_sweep \
        --sweep-2d \
        --widths 256 \
        --lr-values 0.005,0.01,0.02,0.03,0.05 \
        --wd-values 0.0,0.1,0.2,0.3,0.4,0.6 \
        --batch-size 32 \
        --grad-accum-steps 8 \
        --shard $i/5 \
        --save-dir /tmp/sweep_shard_$i \
        --no-plot &
done
echo "All shards launched"
wait
echo "All shards done"
