#!/usr/bin/env bash

download=
prepare=
train=
evaluate=

# download
data_dir=../../data/librispeech
# prepare
out_dir=result
# train
train_json_path=${data_dir}/processed/train.json
valid_json_path=${data_dir}/processed/valid.json
tokenizer_path=${out_dir}/tokenizer.model
checkpoint_path=
# evaluate
test_json_path=${data_dir}/processed/test-clean.json
model_path=${out_dir}/best_model.pt

if [ ${download} ]; then
  echo "Data Download"
  ./download.sh ${data_dir}
fi

if [ ${prepare} ]; then
  echo "Data Preparation"
  python ./preprocess.py --data_dir ${data_dir} --out_dir ${out_dir}
fi

if [ ${train} ]; then
  echo "Model Training"
  python ../../neural_networks/tasks/asr/train.py \
      dataset.train_json_path=${train_json_path} \
      dataset.valid_json_path=${valid_json_path} \
      tokenizer.model_path=${tokenizer_path} \
      trainer.out_dir=${out_dir} \
      trainer.checkpoint_path=${checkpoint_path}
fi

if [ ${evaluate} ]; then
  echo "Model Evaluation"
  python ../../neural_networks/tasks/asr/evaluate.py \
      dataset.test_json_path=${test_json_path} \
      tokenizer.model_path=${tokenizer_path} \
      decode.out_dir=${out_dir} \
      decode.model_path=${model_path}
fi
