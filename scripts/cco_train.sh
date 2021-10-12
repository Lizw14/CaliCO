CKPT_ID=cco
CKPT_ID=tmp
echo ${CKPT_ID}
PROJECT_PATH=.
CKPT_PATH=${PROJECT_PATH}/ckpt/${CKPT_ID}
if [ -d ${CKPT_PATH} ]; then
  echo ${CKPT_PATH} exists!
  #exit
  rm -r ${CKPT_PATH}
fi
mkdir -p ${CKPT_PATH}/bkup
cp $0 ${CKPT_PATH}/bkup
cp ${PROJECT_PATH}/scripts/train_main.py ${CKPT_PATH}/bkup/
cp ${PROJECT_PATH}/core/models/symbolic.py ${CKPT_PATH}/bkup/
cp ${PROJECT_PATH}/core/models/concept_embedding.py ${CKPT_PATH}/bkup/
cp ${PROJECT_PATH}/core/models/model.py ${CKPT_PATH}/bkup/
cp ${PROJECT_PATH}/core/datasets/datasets.py ${CKPT_PATH}/bkup/
cd ${PROJECT_PATH}

python scripts/train_main.py \
  --ckpt_path ${CKPT_PATH} \
  --num_worker 8 \
  --batch_size 256 \
  --lr 5e-4 \
  --num_train_epoches 30 \
  --log_every 20 \
  --eval_every 1 \
  --ckpt_every 1 \
  --hidden_dims 300 300 \
  --test_split testdev \
  --warmup_steps 1000 \
  --train_question_json_path data/orig_data/questions1.2/train_balanced_questions.json,data/orig_data/questions1.2/val_balanced_questions.json \
  --val_question_json_path data/orig_data/questions1.2/testdev_balanced_questions.json \
  --val_scenegraph_json_path data/orig_data/sceneGraphs/val_sceneGraphs.json \
  --info_json_path data/features/sgg_info.json \
  --objects_h5_path data/features/sgg_features.h5 \
  --input_dims 2048 2048 \
  --sgg_vocab_pth data//gqa_vocab_taxo.json \


