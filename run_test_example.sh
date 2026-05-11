python test_example.py --config cfg/train.yaml \
    --resume checkpoints/tracegen_model.pth \
    --resume checkpoints/tracegen_model.pth \
    --test /n/home06/jcheng65/workspace/TraceGen/sample_test \
    --output ./test_result \
    --guidance_scale 1.0 \
    --num_samples_per_scale 1 \
    --override \
        model.decoder.num_layers=6 \
        model.decoder.num_attention_heads=12 \
        model.decoder.latent_dim=768