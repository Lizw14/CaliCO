CKPT_ID=cco_trained

PROJECT_PATH=.
CKPT_PATH=${PROJECT_PATH}/ckpt/${CKPT_ID}
if [ -d ${CKPT_PATH} ]; then
  echo testing ${CKPT_PATH}
fi
cd ${PROJECT_PATH}

python scripts/train_main.py \
  --ckpt_path ${CKPT_PATH} \
  --num_worker 0 \
  --batch_size 1024 \
  --is_test True \
  --test_split testdev \
  --hidden_dims 300 300 \
  --val_question_json_path data/orig_data/questions1.2/testdev_balanced_questions.json \
  --info_json_path data/features/sgg_info.json \
  --objects_h5_path data/features/sgg_features.h5 \
  --sgg_vocab_pth data/gqa_vocab_taxo.json
