if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=96
model_name=Paeformer
root_path=./dataset/
data_path=ETTm1.csv
data_name=ETTm1
features=M
enc_in=7
des=Exp
itr=1
batch_size=32

python -u run_longExp.py \
    --is_training 1 \
    --alpha 3 \
    --slice_len 8 \
    --middle_len 256 \
    --hidden_len 64 \
    --use_norm 0 \
    --n_heads 4 \
    --slice_stride 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id ETTm1_96_96\
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len 96 \
    --enc_in $enc_in \
    --des $des \
    --itr $itr \
    --batch_size $batch_size \
    --learning_rate 0.001 > logs/LongForecasting/${model_name}_ETTm1_96_96.log

python -u run_longExp.py \
    --is_training 1 \
    --alpha 3 \
    --slice_len 8 \
    --n_heads 4 \
    --middle_len 256 \
    --hidden_len 32 \
    --use_norm 0 \
    --gpu 2 \
    --slice_stride 8 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id ETTm1_96_96\
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len 192 \
    --enc_in $enc_in \
    --des $des \
    --itr $itr \
    --batch_size $batch_size \
    --learning_rate 0.0001 > logs/LongForecasting/${model_name}_ETTm1_96_192.log

python -u run_longExp.py \
    --is_training 1 \
    --alpha 3 \
    --slice_len 8 \
    --middle_len 256 \
    --gpu 2 \
    --hidden_len 32 \
    --use_norm 0 \
    --n_heads 4 \
    --slice_stride 8 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id ETTm1_96_96\
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len 336 \
    --enc_in $enc_in \
    --des $des \
    --itr $itr \
    --batch_size $batch_size \
    --learning_rate 0.001 > logs/LongForecasting/${model_name}_ETTm1_96_336.log

python -u run_longExp.py \
    --is_training 1 \
    --alpha 3 \
    --slice_len 8 \
    --middle_len 256 \
    --gpu 2 \
    --use_norm 0 \
    --hidden_len 32 \
    --n_heads 4 \
    --slice_stride 8 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id ETTm1_96_96\
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len 720 \
    --enc_in $enc_in \
    --des $des \
    --itr $itr \
    --batch_size $batch_size \
    --learning_rate 0.001 > logs/LongForecasting/${model_name}_ETTm1_96_720.log




