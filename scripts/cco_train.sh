CKPT_ID=cco_train_hier
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
  --info_json_path /data/c/zhuowan/gqa_project/Scene-Graph-Benchmark.pytorch/checkpoints/GQA_pretrained_rcnn_attr_taxo_ce/inference/trainall/sgg_info.json \
  --objects_h5_path /home/zhuowan/data/sgg_features.h5 \
  --input_dims 2048 2048 \
  --sgg_vocab_pth /home/zhuowan/zhuowan/gqa_project/Scene-Graph-Benchmark.pytorch/datasets/GQA/gqa_vocab_taxo.json \


  # --pretrained_path ckpt/Feb_ft2_mergeadd_weightalllstm_picklocw_box5_prep \


  # --val_question_json_path data/orig_data/questions1.2/testdev_balanced_questions.json,data/orig_data/questions1.2/val_balanced_questions.json \
  # --train_question_json_path data/elias/train_balanced_questions_new.json \
  # --train_question_json_path data/orig_data/questions1.2/train_balanced_questions.json,data/orig_data/questions1.2/val_balanced_questions.json \

#
  # --dataset_cache train10k



  
# /home/zhuowan/anaconda2/envs/py36/bin/python scripts/gqa_nscl_train.py \
#   --ckpt_path ${CKPT_PATH} \
#   --num_worker 8 \
#   --batch_size 256 \
#   --lr 5e-4 \
#   --num_train_epoches 5 \
#   --log_every 20 \
#   --eval_every 1 \
#   --ckpt_every 1 \
#   --hidden_dims 300 300 \
#   --test_split testdev \
#   --warmup_steps 1000 \
#   --num_worker 4 \
#   --train_question_json_path data/orig_data/questions1.2/train_all_questions/train_all_questions_2.json,data/orig_data/questions1.2/train_all_questions/train_all_questions_3.json,data/orig_data/questions1.2/train_all_questions/train_all_questions_8.json,data/orig_data/questions1.2/train_all_questions/train_all_questions_4.json,data/orig_data/questions1.2/train_all_questions/train_all_questions_6.json,data/orig_data/questions1.2/train_all_questions/train_all_questions_5.json,data/orig_data/questions1.2/train_all_questions/train_all_questions_7.json,data/orig_data/questions1.2/train_all_questions/train_all_questions_1.json,data/orig_data/questions1.2/train_all_questions/train_all_questions_9.json,data/orig_data/questions1.2/train_all_questions/train_all_questions_0.json \
#   --val_question_json_path data/orig_data/questions1.2/testdev_balanced_questions.json \
#   --info_json_path /data/c/zhuowan/gqa_project/Scene-Graph-Benchmark.pytorch/checkpoints/GQA_pretrained_rcnn_attr_taxo_ce/inference/trainall/sgg_info.json \
#   --objects_h5_path /home/zhuowan/data/sgg_features.h5 \
#   --input_dims 2048 2048 \
#   --sgg_vocab_pth /home/zhuowan/zhuowan/gqa_project/Scene-Graph-Benchmark.pytorch/datasets/GQA/gqa_vocab_taxo.json \


  # --dataset_cache train10k




#| tee ${CKPT_PATH}/log.txt
  
  
  #--dataset_cache train10k
  
  #--gdef_pth /home/yixiao/Scene-Graph-Benchmark.pytorch/datasets/GQA/taxonomies.json


#  --sgg_vocab_pth /home/zhuowan/zhuowan/gqa_project/Scene-Graph-Benchmark.pytorch/datasets/GQA/gqa_vocab_taxo.json

#  --info_json_path /data/c/zhuowan/gqa_project/Scene-Graph-Benchmark.pytorch/checkpoints/GQA_pretrained_rcnn_attr_taxo_bce/inference/train10k/sgg_info.json \
#  --objects_h5_path /data/c/zhuowan/gqa_project/Scene-Graph-Benchmark.pytorch/checkpoints/GQA_pretrained_rcnn_attr_taxo_bce/inference/train10k/sgg_features.h5 \
#  --info_json_path /data/c/zhuowan/gqa_project/Scene-Graph-Benchmark.pytorch/checkpoints/GQA_pretrained_rcnn_attr_thres1/inference/all_100thres0.2/sgg_info.json \
#  --objects_h5_path /data/c/zhuowan/gqa_project/Scene-Graph-Benchmark.pytorch/checkpoints/GQA_pretrained_rcnn_attr_thres1/inference/all_100thres0.2/sgg_features.h5 \
#  --input_dims 2048 2048 \
  

#  --pretrained_path ckpt/gqa_nscl_glovecorrect_ver12_allsymbolonlycorrect_embedsim


#  --dataset_cache train10k \

# 4352 3733 2667
#  --val_question_json_path data/orig_data/questions1.3/val_balanced_questions.json \
#  --is_gtencode True \

#  --pretrained_path ckpt/gqa_nscl_fq_gtencode_prepspatialcat \

#  --val_scenegraph_json_path data/orig_data/sceneGraphs/train_sceneGraphs.json \
#  --dataset_cache relleftright 





#| tee ${CKPT_PATH}/log.txt

#  --dataset_cache origin:op-type-[allowed={filter,query}] \
#  --dataset_cache catdoghorse \
#  --val_question_json_path data/orig_data/questions1.3/train_balanced_questions.json
