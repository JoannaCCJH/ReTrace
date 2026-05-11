xvfb-run -a python -m sim.run_libero_eval \
    --config cfg/train.yaml \
    --resume checkpoints/tracegen_model.pth \
    --num_trials 5 \
    --output ./libero_results/libero_90/run8 \
    --override model.decoder.num_attention_heads=12 model.decoder.num_layers=6  \
    --benchmark libero_90 \
    --task STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy \
    --object_obs_key porcelain_mug_1_pos \
    --goal_site_name study_table_desk_caddy_right_region \
    --dz_scale 5.5 \
    --placement_mode descend \

xvfb-run -a python -m sim.run_libero_eval \
      --config cfg/train.yaml \
      --resume checkpoints/tracegen_model.pth \
      --num_trials 5 \
      --output ./libero_results/libero_90/yellow_white_mug_caddy \
      --override model.decoder.num_attention_heads=12 model.decoder.num_layers=6 \
      --benchmark libero_90 \
      --task STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy \
      --object_obs_key white_yellow_mug_1_pos \
      --goal_site_name study_table_desk_caddy_right_region \
      --dz_scale 5.5 \
      --placement_mode trace


xvfb-run -a python -m sim.run_libero_eval \
      --config cfg/train.yaml \
      --resume checkpoints/tracegen_model.pth \
      --num_trials 5 \
      --output ./libero_results/libero_90/replan100 \
      --override model.decoder.num_attention_heads=12 model.decoder.num_layers=6 \
      --benchmark libero_90 \
      --task STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy \
      --object_obs_key porcelain_mug_1_pos \
      --goal_site_name study_table_desk_caddy_right_region \
      --dz_scale 5.5 \
      --placement_mode descend \
      --replan_freq 100 \
      --viz_replans all

xvfb-run -a python -m sim.visualize_pred

xvfb-run -a python -m sim.visualize_pred \
    --config /n/home06/jcheng65/workspace/TraceGen/checkpoints/train.yaml \
    --resume /n/home06/jcheng65/workspace/TraceGen/checkpoints/tracegen_model.pth \
    --benchmark libero_90