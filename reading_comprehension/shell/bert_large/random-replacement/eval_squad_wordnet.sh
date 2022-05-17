#!/bin/bash
# ==============================================================================
# Copyright 2019 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LC_CTYPE=en_US.UTF-8

model_name=bert_large
BERT_DIR=uncased_L-24_H-1024_A-16

log_dir=log_$model_name/random-replacement
output_dir=output_$model_name/random-replacement

if [ ! -d $log_dir ]; then
mkdir -p $log_dir
fi

if [ ! -d $output_dir ]; then
mkdir -p $output_dir
fi

export FLAGS_cudnn_deterministic=true
export FLAGS_cpu_deterministic=true

PWD_DIR=`pwd`
DATA=../data/
CPT_EMBEDDING_PATH=../retrieve_concepts/KB_embeddings/wn_concept2vec.txt

CKPT_DIR=$1

nohup python3 src/run_squad.py \
  --batch_size 6 \
  --do_train false \
  --do_predict true \
  --use_ema false \
  --do_lower_case true \
  --init_pretraining_params $BERT_DIR/params \
  --init_checkpoint $CKPT_DIR \
  --train_file $DATA/SQuAD/train-v2.0.json \
  --predict_file $DATA/SQuAD/random-replacement/dev-v2.0.json \
  --vocab_path $BERT_DIR/vocab.txt \
  --bert_config_path $BERT_DIR/bert_config.json \
  --freeze false \
  --max_seq_len 384 \
  --doc_stride 128 \
  --concept_embedding_path $CPT_EMBEDDING_PATH \
  --random_seed 45 \
  --version_2_with_negative true \
  --random_replacement true \
  --checkpoints $output_dir/ 1>$PWD_DIR/$log_dir/eval.log 2>&1 &
