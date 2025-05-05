if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=96
model_name=Paeformer
root_path=./dataset/solar/
data_path=solar_AL.txt
data_name=Solar
features=M
enc_in=137
des=Exp
itr=1
batch_size=32


python -u run_longExp.py \
    --is_training 1 \
    --alpha 1 \
    --slice_len 24 \
    --middle_len 512 \
    --hidden_len 512 \
    --slice_stride 24 \
    --gpu 2 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id solar_96_96\
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len 96 \
    --enc_in $enc_in \
    --des $des \
    --itr $itr \
    --batch_size $batch_size \
    --learning_rate 0.0005 > logs/LongForecasting/${model_name}_solar_96_96.log

python -u run_longExp.py \
    --is_training 1 \
    --alpha 1 \
    --slice_len 24 \
    --e_layers 4 \
    --middle_len 512 \
    --hidden_len 512 \
    --slice_stride 24 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id solar_96_96\
    --model $model_name \
    --data $data_name \
    --gpu 2 \
    --features $features \
    --seq_len $seq_len \
    --pred_len 192 \
    --enc_in $enc_in \
    --des $des \
    --itr $itr \
    --batch_size $batch_size \
    --learning_rate 0.0005 > logs/LongForecasting/${model_name}_solar_96_192.log

python -u run_longExp.py \
    --is_training 1 \
    --alpha 1 \
    --slice_len 24 \
    --middle_len 512 \
    --hidden_len 512 \
    --slice_stride 24 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id solar_96_96\
    --model $model_name \
    --data $data_name \
    --gpu 2 \
    --features $features \
    --seq_len $seq_len \
    --pred_len 336 \
    --enc_in $enc_in \
    --des $des \
    --itr $itr \
    --batch_size $batch_size \
    --learning_rate 0.0005 > logs/LongForecasting/${model_name}_solar_96_336.log

python -u run_longExp.py \
    --is_training 1 \
    --alpha 1 \
    --e_layers 4 \
    --slice_len 48 \
    --middle_len 512 \
    --hidden_len 512 \
    --slice_stride 48 \
    --gpu 2 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id solar_96_96\
    --model $model_name \
    --data $data_name \
    --features $features \
    --seq_len $seq_len \
    --pred_len 720 \
    --enc_in $enc_in \
    --des $des \
    --itr $itr \
    --batch_size $batch_size \
    --learning_rate 0.0005 > logs/LongForecasting/${model_name}_solar_96_720.log