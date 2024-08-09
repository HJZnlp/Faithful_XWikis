# Faithful_XWikis

This repository contains the code and data for ACL2024 paper: Leveraging Entailment Judgements in Cross-Lingual Summarisation [Paper](https://arxiv.org/pdf/2408.00675).

Our code is based on a clone from 14 July 2020 of the [Fairseq](https://github.com/pytorch/fairseq) Library. The XWikis corpus extended with the Chinese language can be found on HuggingFace datasets, follow this [link](https://huggingface.co/datasets/GEM/xwikis) 

Should you have any queries please contact me at v1hzha17@ed.ac.uk


## Quickstart
```
git clone https://github.com/HJZnlp/Faithful_XWikis.git
cd Faithful_XWikis
pip install --editable ./
```
## Training
To train the model, use the following commands. Replace the placeholder paths with your actual paths:
```

PRETRAIN=PATH_TO_MBART
BPE_DATA=PATH_TO_MBART_BPE

langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN,af_ZA,az_AZ,bn_IN,fa_IR,he_IL,hr_HR,id_ID,ka_GE,km_KH,mk_MK,ml_IN,mn_MN,mr_IN,pl_PL,ps_AF,pt_XX,sv_SE,sw_KE,ta_IN,te_IN,th_TH,tl_XX,uk_UA,ur_PK,xh_ZA,gl_ES,sl_SI
MAX=1024

EXEC_DATA=YOUR_BIN_DATA
noise_data=YOUR_BIN_NOISE_LABEL
MODEL_PATH=PATH_TO_OUTPUT

SRC=SOURCE_LANGE # fr_XX, de_DE, zh_CN or cs_CZ
TGT=en_XX
max_tokens=1024

mkdir $MODEL_PATH


fairseq-train $EXEC_DATA \
  --encoder-normalize-before --decoder-normalize-before \
  --arch mbart_large --layernorm-embedding \
  --task sum_from_pretrained_bart \
  --source-lang $SRC --target-lang $TGT \
  --criterion label_smoothed_cross_entropy_ul --label-smoothing 0.2 \
  --dataset-impl mmap \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 --warmup-updates 2500 --max-update 20000  --total-num-update 20000 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens $max_tokens --update-freq 40 \
  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --seed 222 --log-format simple --log-interval 20 \
  --restore-file $PRETRAIN \
  --langs $langs \
  --ddp-backend no_c10d \
  --save-dir $MODEL_PATH \
  --skip-invalid-size-inputs-valid-test \
  --num-workers 10 \
  --truncate-source --max-source-positions $MAX --max-target-positions $MAX \
  --find-unused-parameters \
  --memory-efficient-fp16 --weight-decay 0.01 --clip-norm 0.1 \
  --prepend-bos --blocks --load-noise-labels --noise-path $noise_data

```
## Hyperparameters

| Parameter        | Value  |
|------------------|--------|
| Batch size       | 64     |
| Total updates    | 30K    |
| Dropout          | 0.3    |


## Inference
To run inference, use the following commands. Again, replace the placeholder paths with your actual paths:
```

EVAL_SET=test
MAXTOKENS=1024
MAXLENB=150
MINLEN=50
LENPEN=2

RESULTS_PATH=PATH_TO_MODEL

fairseq-generate $EXEC_DATA \
  --path $MODEL_PATH/checkpoint_best.pt \
  --task sum_from_pretrained_bart \
  --gen-subset $EVAL_SET \
  --source-lang $SRC --target-lang $TGT \
  --bpe 'sentencepiece' --sentencepiece-vocab $BPE_DATA/sentence.bpe.model \
  --skip-invalid-size-inputs-valid-test \
  --max-len-b $MAXLENB --beam 5 --min-len $MINLEN --lenpen $LENPEN --no-repeat-ngram-size 3 \
  --sacrebleu --compute-rouge \
  --max-sentences 20 --langs $langs \
  --results-path $RESULTS_PATH \
  --truncate-source --max-source-positions $MAXTOKENS --max-target-positions $MAXTOKENS \
  --memory-efficient-fp16 \
  --prepend-bos \
  --blocks \
  --model-overrides '{"load_checkpoint_mtasks":False}' --load-noise-labels

```

## Evaluation
1. [INFUSE](https://github.com/HJZnlp/infuse/tree/main)
2. [LABSE](https://github.com/yang-zhang/labse-pytorch)
3. [UNIEVAL](https://github.com/maszhongming/UniEval)
Note we use unieval-sum for faithfulness evaluation.  


## Citation
If you find this repository useful in your research, please consider citing our paper:

```
@misc{zhang2024leveragingentailmentjudgementscrosslingual,
      title={Leveraging Entailment Judgements in Cross-Lingual Summarisation}, 
      author={Huajian Zhang and Laura Perez-Beltrachini},
      year={2024},
      eprint={2408.00675},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.00675}, 
}
```
