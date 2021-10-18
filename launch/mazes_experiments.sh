#dt_net_2d
python train_model.py --alpha 0.00 --epochs 50 --lr 0.001 --lr_decay step --lr_factor 0.1 --lr_schedule 100 --lr_throttle --max_iters 30 --model dt_net_2d --optimizer adam --output mazes_experiments --problem mazes --test_batch_size 25 --test_data 33 --test_iterations 20 30 40 50 60 70 80 90 100 150 200 300 400 500 750 1000 --test_mode max_conf --train_batch_size 50 --train_data 9 --train_log mazes_experiment_1.txt --train_mode progressive --val_period 20 --warmup_period 10 --width 128

python train_model.py --alpha 0.01 --epochs 50 --lr 0.001 --lr_decay step --lr_factor 0.1 --lr_schedule 100 --lr_throttle --max_iters 30 --model dt_net_2d --optimizer adam --output mazes_experiments --problem mazes --test_batch_size 25 --test_data 33 --test_iterations 20 30 40 50 60 70 80 90 100 150 200 300 400 500 750 1000 --test_mode max_conf --train_batch_size 50 --train_data 9 --train_log mazes_experiment_2.txt --train_mode progressive --val_period 20 --warmup_period 10 --width 128

python train_model.py --alpha 0.50 --epochs 50 --lr 0.001 --lr_decay step --lr_factor 0.1 --lr_schedule 100 --lr_throttle --max_iters 30 --model dt_net_2d --optimizer adam --output mazes_experiments --problem mazes --test_batch_size 25 --test_data 33 --test_iterations 20 30 40 50 60 70 80 90 100 150 200 300 400 500 750 1000 --test_mode max_conf --train_batch_size 50 --train_data 9 --train_log mazes_experiment_3.txt --train_mode progressive --val_period 20 --warmup_period 10 --width 128

python train_model.py --alpha 1.00 --epochs 50 --lr 0.001 --lr_decay step --lr_factor 0.1 --lr_schedule 100 --lr_throttle --max_iters 30 --model dt_net_2d --optimizer adam --output mazes_experiments --problem mazes --test_batch_size 25 --test_data 33 --test_iterations 20 30 40 50 60 70 80 90 100 150 200 300 400 500 750 1000 --test_mode max_conf --train_batch_size 50 --train_data 9 --train_log mazes_experiment_4.txt --train_mode progressive --val_period 20 --warmup_period 10 --width 128


#dt_net_recallx_2d
python train_model.py --alpha 0.00 --epochs 50 --lr 0.001 --lr_decay step --lr_factor 0.1 --lr_schedule 100 --lr_throttle --max_iters 30 --model dt_net_recallx_2d --optimizer adam --output mazes_experiments --problem mazes --test_batch_size 25 --test_data 33 --test_iterations 20 30 40 50 60 70 80 90 100 150 200 300 400 500 750 1000 --test_mode max_conf --train_batch_size 50 --train_data 9 --train_log mazes_experiment_5.txt --train_mode progressive --val_period 20 --warmup_period 10 --width 128

python train_model.py --alpha 0.01 --epochs 50 --lr 0.001 --lr_decay step --lr_factor 0.1 --lr_schedule 100 --lr_throttle --max_iters 30 --model dt_net_recallx_2d --optimizer adam --output mazes_experiments --problem mazes --test_batch_size 25 --test_data 33 --test_iterations 20 30 40 50 60 70 80 90 100 150 200 300 400 500 750 1000 --test_mode max_conf --train_batch_size 50 --train_data 9 --train_log mazes_experiment_6.txt --train_mode progressive --val_period 20 --warmup_period 10 --width 128


#feedforward
python train_model.py --alpha 0.00 --epochs 200 --lr 0.0001 --lr_decay step --lr_factor 0.1 --lr_schedule 175 --max_iters 30 --model feedforward_net_2d --optimizer adam --output mazes_experiments --problem mazes --test_batch_size 25 --test_data 33 --test_iterations 30 --test_mode default --train_batch_size 50 --train_data 9 --train_log mazes_experiment_7.txt --train_mode progressive --val_period 50 --warmup_period 10 --width 128

python train_model.py --alpha 0.50 --epochs 200 --lr 0.0001 --lr_decay step --lr_factor 0.1 --lr_schedule 175 --max_iters 30 --model feedforward_net_2d --optimizer adam --output mazes_experiments --problem mazes --test_batch_size 25 --test_data 33 --test_iterations 30 --test_mode default --train_batch_size 50 --train_data 9 --train_log mazes_experiment_8.txt --train_mode progressive --val_period 50 --warmup_period 10 --width 128


#feedforward recallx
python train_model.py --alpha 0.00 --epochs 200 --lr 0.0001 --lr_decay step --lr_factor 0.1 --lr_schedule 175 --max_iters 30 --model feedforward_net_recallx_2d --optimizer adam --output mazes_experiments --problem mazes --test_batch_size 25 --test_data 33 --test_iterations 30 --test_mode default --train_batch_size 50 --train_data 9 --train_log mazes_experiment_8.txt --train_mode progressive --val_period 20 --warmup_period 10 --width 128

python train_model.py --alpha 0.50 --epochs 200 --lr 0.0001 --lr_decay step --lr_factor 0.1 --lr_schedule 175 --max_iters 30 --model feedforward_net_recallx_2d --optimizer adam --output mazes_experiments --problem mazes --test_batch_size 25 --test_data 33 --test_iterations 30 --test_mode default --train_batch_size 50 --train_data 9 --train_log mazes_experiment_9.txt --train_mode progressive --val_period 20 --warmup_period 10 --width 128
