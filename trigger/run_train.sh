python -m trigger.train \
    --data-dir trigger/data \
    --out trigger/checkpoints/trigger.pt \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-3 \
    --weight-decay 5e-3 \
    --seed 0
