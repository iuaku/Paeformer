if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=96
model_name=Paeformer
root_path=./dataset/
data_path=electricity.csv
data_name=custom
features=M
enc_in=321
des=Exp
itr=1
batch_size=16

python -u run_longExp.py \
    --is_training 1 \
    --alpha 1 \
    --slice_len 48 \
    --middle_len 256 \
    --hidden_len 256 \
    --slice_stride 48 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id electricity_96_96\
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len 96 \
    --enc_in $enc_in \
    --des $des \
    --gpu 0 \
    --itr $itr \
    --batch_size $batch_size \
    --learning_rate 0.001   > logs/LongForecasting/${model_name}_electricity_${seq_len}_96.log



python -u run_longExp.py \
    --is_training 1 \
    --alpha 1 \
    --slice_len 48 \
    --e_layers 4 \
    --middle_len 256 \
    --hidden_len 256 \
    --slice_stride 48 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id electricity_96_192\
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len 192 \
    --gpu 0 \
    --enc_in $enc_in \
    --des $des \
    --itr $itr \
    --batch_size $batch_size \
    --learning_rate 0.001 > logs/LongForecasting/${model_name}_electricity_96_192.log

python -u run_longExp.py \
    --is_training 1 \
    --alpha 1 \
    --slice_len 48 \
    --middle_len 512 \
    --hidden_len 512 \
    --gpu 0 \
    --slice_stride 48 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id electricity_96_336\
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len 336 \
    --enc_in $enc_in \
    --des $des \
    --itr $itr \
    --batch_size $batch_size \
    --learning_rate 0.001 > logs/LongForecasting/${model_name}_electricity_96_336.log

python -u run_longExp.py \
    --is_training 1 \
    --alpha 1 \
    --slice_len 48 \
    --middle_len 512 \
    --hidden_len 512 \
    --gpu 0 \
    --slice_stride 48 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id electricity_96_720\
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len 720 \
    --enc_in $enc_in \
    --des $des \
    --itr $itr \
    --batch_size $batch_size \
    --learning_rate 0.001 > logs/LongForecasting/${model_name}_electricity_96_720.log
