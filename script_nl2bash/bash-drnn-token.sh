#!/bin/bash

# reproduce experiments using seq2seq with attention model on the bash dataset

ARGS=${@:1}

./bash-run.sh \
    --dataset bash \
    --channel token \
    --decoder_topology doublernn \
    --token_decoding_algorithm greedy \
    --batch_size 64 \
    --sc_token_dim 200 \
    --learning_rate 0.0001 \
    --steps_per_epoch 4000 \
    --notg_token_use_attention \
    --nouse_copy \
    --tg_token_attn_fun non-linear \
    --universal_keep 0.6 \
    --sc_input_keep 1.0 \
    --tg_input_keep 1.0 \
    --sc_output_keep 1.0 \
    --tg_output_keep 1.0 \
    --attention_input_keep 1.0 \
    --attention_output_keep 1.0 \
    --beta 0 \
    --create_fresh_params \
    --min_vocab_frequency 4 \
    ${ARGS}
