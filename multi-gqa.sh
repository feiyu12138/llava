cur=/home/jchen293/llava_git/llava
bash scripts/v1_5/eval/gqa_stride_8_layer_16_grouping_1d_v2.sh
cd $cur
bash scripts/v1_5/eval/gqa_stride_8_layer_16_grouping_1d_v3.sh
cd $cur
bash scripts/v1_5/eval/gqa_stride_16_layer_16_grouping_1d_v3.sh
cd $cur
bash scripts/v1_5/eval/gqa_stride_64_layer_16_grouping_1d_v3.sh