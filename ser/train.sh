CUDA_IDX=0

CUDA_VISIBLE_DEVICES=$CUDA_IDX python train.py \
    --ssl_model_name facebook/wav2vec2-base \
    --num_samples_per_class 1000

CUDA_VISIBLE_DEVICES=$CUDA_IDX python train.py \
    --ssl_model_name microsoft/wavlm-base-plus \
    --num_samples_per_class 1000

CUDA_VISIBLE_DEVICES=$CUDA_IDX python train.py \
    --ssl_model_name facebook/hubert-base-ls960 \
    --num_samples_per_class 1000