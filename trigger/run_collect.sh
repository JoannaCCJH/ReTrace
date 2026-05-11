#!/usr/bin/env bash
# Collect counterfactual replan-trigger training data for two LIBERO-90 mug
# tasks, then aggregate into train.npz / val.npz under trigger/data/.

set -euo pipefail

CONFIG=${CONFIG:-cfg/train.yaml}
RESUME=${RESUME:-checkpoints/tracegen_model.pth}
DZ=${DZ:-5.5}
RAW=${RAW:-trigger/data/raw}
NUM_EPISODES=${NUM_EPISODES:-110}
NUM_NOMINAL=${NUM_NOMINAL:-55}
K=${K:-5}
R_CLOSE=${R_CLOSE:-0.05}

# Task 1: white mug -> right of caddy (placement_mode=descend, mirrors run_sim.sh).
xvfb-run -a python -m trigger.collect \
    --config "$CONFIG" --resume "$RESUME" \
    --override model.decoder.num_attention_heads=12 model.decoder.num_layers=6 \
    --benchmark libero_90 \
    --task STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy \
    --object_obs_key porcelain_mug_1_pos \
    --perturb_body_name porcelain_mug_1_main \
    --goal_site_name study_table_desk_caddy_right_region \
    --dz_scale "$DZ" --placement_mode descend \
    --output "$RAW/STUDY_SCENE3_white_mug" \
    --num_episodes "$NUM_EPISODES" --num_nominal "$NUM_NOMINAL" \
    --K "$K" --R_close "$R_CLOSE"

# Task 2: yellow-and-white mug -> right of caddy (placement_mode=trace per run_sim.sh).
xvfb-run -a python -m trigger.collect \
    --config "$CONFIG" --resume "$RESUME" \
    --override model.decoder.num_attention_heads=12 model.decoder.num_layers=6 \
    --benchmark libero_90 \
    --task STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy \
    --object_obs_key white_yellow_mug_1_pos \
    --perturb_body_name white_yellow_mug_1_main \
    --goal_site_name study_table_desk_caddy_right_region \
    --dz_scale "$DZ" --placement_mode trace \
    --output "$RAW/STUDY_SCENE1_yellow_white_mug" \
    --num_episodes "$NUM_EPISODES" --num_nominal "$NUM_NOMINAL" \
    --K "$K" --R_close "$R_CLOSE"

# Aggregate both tasks into trigger/data/train.npz and val.npz.
python -m trigger.build_splits \
    --raw_dirs "$RAW/STUDY_SCENE3_white_mug" \
               "$RAW/STUDY_SCENE1_yellow_white_mug" \
    --out_dir trigger/data \
    --train_episodes_per_task 100 --val_episodes_per_task 10
