python train_model.py problem.hyp.alpha=0.00 problem/model=dt_net_2d problem=mazes name=mazes_oldthrottle +problem.hyp.old_throttle=True problem.hyp.lr_throttle=False
python train_model.py problem.hyp.alpha=0.01 problem/model=dt_net_2d problem=mazes name=mazes_oldthrottle +problem.hyp.old_throttle=True problem.hyp.lr_throttle=False
python train_model.py problem.hyp.alpha=0.50 problem/model=dt_net_2d problem=mazes name=mazes_oldthrottle +problem.hyp.old_throttle=True problem.hyp.lr_throttle=False
python train_model.py problem.hyp.alpha=1.00 problem/model=dt_net_2d problem=mazes name=mazes_oldthrottle +problem.hyp.old_throttle=True problem.hyp.lr_throttle=False
python train_model.py problem.hyp.alpha=0.00 problem/model=dt_net_recall_2d problem=mazes name=mazes_oldthrottle +problem.hyp.old_throttle=True problem.hyp.lr_throttle=False
python train_model.py problem.hyp.alpha=0.01 problem/model=dt_net_recall_2d problem=mazes name=mazes_oldthrottle +problem.hyp.old_throttle=True problem.hyp.lr_throttle=False
python train_model.py problem.hyp.alpha=0.50 problem/model=dt_net_recall_2d problem=mazes name=mazes_oldthrottle +problem.hyp.old_throttle=True problem.hyp.lr_throttle=False
python train_model.py problem.hyp.alpha=1.00 problem/model=dt_net_recall_2d problem=mazes name=mazes_oldthrottle +problem.hyp.old_throttle=True problem.hyp.lr_throttle=False
python train_model.py problem.hyp.alpha=0.00 problem.hyp.epochs=200 problem.hyp.lr=0.0001 problem.hyp.lr_schedule=[175] problem/model=ff_net_2d problem=mazes name=mazes_ablation
python train_model.py problem.hyp.alpha=0.01 problem.hyp.epochs=200 problem.hyp.lr=0.0001 problem.hyp.lr_schedule=[175] problem/model=ff_net_2d problem=mazes name=mazes_ablation
python train_model.py problem.hyp.alpha=0.50 problem.hyp.epochs=200 problem.hyp.lr=0.0001 problem.hyp.lr_schedule=[175] problem/model=ff_net_2d problem=mazes name=mazes_ablation
python train_model.py problem.hyp.alpha=0.00 problem.hyp.epochs=200 problem.hyp.lr=0.0001 problem.hyp.lr_schedule=[175] problem/model=ff_net_recall_2d problem=mazes name=mazes_ablation
python train_model.py problem.hyp.alpha=0.01 problem.hyp.epochs=200 problem.hyp.lr=0.0001 problem.hyp.lr_schedule=[175] problem/model=ff_net_recall_2d problem=mazes name=mazes_ablation
python train_model.py problem.hyp.alpha=0.50 problem.hyp.epochs=200 problem.hyp.lr=0.0001 problem.hyp.lr_schedule=[175] problem/model=ff_net_recall_2d problem=mazes name=mazes_ablation


python train_model.py problem.hyp.warmup_period=25 problem.hyp.lr=0.0001 problem.hyp.alpha=0.00 problem/model=dt_net_2d problem=mazes name=mazes_no_throttle problem.hyp.lr_throttle=False
python train_model.py problem.hyp.warmup_period=25 problem.hyp.lr=0.0001 problem.hyp.alpha=0.01 problem/model=dt_net_2d problem=mazes name=mazes_no_throttle problem.hyp.lr_throttle=False
python train_model.py problem.hyp.warmup_period=25 problem.hyp.lr=0.0001 problem.hyp.alpha=0.50 problem/model=dt_net_2d problem=mazes name=mazes_no_throttle problem.hyp.lr_throttle=False
python train_model.py problem.hyp.warmup_period=25 problem.hyp.lr=0.0001 problem.hyp.alpha=1.00 problem/model=dt_net_2d problem=mazes name=mazes_no_throttle problem.hyp.lr_throttle=False
python train_model.py problem.hyp.warmup_period=25 problem.hyp.lr=0.0001 problem.hyp.alpha=0.00 problem/model=dt_net_recall_2d problem=mazes name=mazes_no_throttle problem.hyp.lr_throttle=False
python train_model.py problem.hyp.warmup_period=25 problem.hyp.lr=0.0001 problem.hyp.alpha=0.01 problem/model=dt_net_recall_2d problem=mazes name=mazes_no_throttle problem.hyp.lr_throttle=False
python train_model.py problem.hyp.warmup_period=25 problem.hyp.lr=0.0001 problem.hyp.alpha=0.50 problem/model=dt_net_recall_2d problem=mazes name=mazes_no_throttle problem.hyp.lr_throttle=False
python train_model.py problem.hyp.warmup_period=25 problem.hyp.lr=0.0001 problem.hyp.alpha=1.00 problem/model=dt_net_recall_2d problem=mazes name=mazes_no_throttle problem.hyp.lr_throttle=False

python train_model.py name=mazes_wide_big_batch problem.model.width=200 problem.hyp.train_batch_size=100 problem.hyp.warmup_period=20 problem.hyp.lr=0.00001 problem.hyp.alpha=0.00 problem/model=dt_net_2d problem=mazes problem.hyp.lr_throttle=False
python train_model.py name=mazes_wide_big_batch problem.model.width=200 problem.hyp.train_batch_size=100 problem.hyp.warmup_period=20 problem.hyp.lr=0.00001 problem.hyp.alpha=0.01 problem/model=dt_net_2d problem=mazes problem.hyp.lr_throttle=False
python train_model.py name=mazes_wide_big_batch problem.model.width=200 problem.hyp.train_batch_size=100 problem.hyp.warmup_period=20 problem.hyp.lr=0.00001 problem.hyp.alpha=0.50 problem/model=dt_net_2d problem=mazes problem.hyp.lr_throttle=False
python train_model.py name=mazes_wide_big_batch problem.model.width=200 problem.hyp.train_batch_size=100 problem.hyp.warmup_period=20 problem.hyp.lr=0.00001 problem.hyp.alpha=1.00 problem/model=dt_net_2d problem=mazes problem.hyp.lr_throttle=False
python train_model.py name=mazes_wide_big_batch problem.model.width=200 problem.hyp.train_batch_size=100 problem.hyp.warmup_period=20 problem.hyp.lr=0.00001 problem.hyp.alpha=0.00 problem/model=dt_net_recall_2d problem=mazes problem.hyp.lr_throttle=False
python train_model.py name=mazes_wide_big_batch problem.model.width=200 problem.hyp.train_batch_size=100 problem.hyp.warmup_period=20 problem.hyp.lr=0.00001 problem.hyp.alpha=0.01 problem/model=dt_net_recall_2d problem=mazes problem.hyp.lr_throttle=False
python train_model.py name=mazes_wide_big_batch problem.model.width=200 problem.hyp.train_batch_size=100 problem.hyp.warmup_period=20 problem.hyp.lr=0.00001 problem.hyp.alpha=0.50 problem/model=dt_net_recall_2d problem=mazes problem.hyp.lr_throttle=False
python train_model.py name=mazes_wide_big_batch problem.model.width=200 problem.hyp.train_batch_size=100 problem.hyp.warmup_period=20 problem.hyp.lr=0.00001 problem.hyp.alpha=1.00 problem/model=dt_net_recall_2d problem=mazes problem.hyp.lr_throttle=False

python train_model.py name=mazes_hyptune problem.hyp.alpha=0.01 problem/model=dt_net_recall_2d problem=mazes problem.hyp.lr_throttle=False +problem.hyp.old_throttle=True
python train_model.py name=mazes_hyptune problem.hyp.train_batch_size=100 problem.hyp.alpha=0.01 problem/model=dt_net_recall_2d problem=mazes problem.hyp.lr_throttle=False +problem.hyp.old_throttle=True
python train_model.py name=mazes_hyptune problem.model.width=200 problem.hyp.alpha=0.01 problem/model=dt_net_recall_2d problem=mazes problem.hyp.lr_throttle=False +problem.hyp.old_throttle=True
python train_model.py name=mazes_hyptune problem.hyp.alpha=0.01 problem/model=dt_net_recall_2d problem=mazes problem.hyp.lr_throttle=False
python train_model.py name=mazes_hyptune problem.hyp.train_batch_size=100 problem.hyp.lr=0.0001 problem.hyp.alpha=0.01 problem/model=dt_net_recall_2d problem=mazes problem.hyp.lr_throttle=False
python train_model.py name=mazes_hyptune problem.hyp.train_batch_size=100 problem.hyp.lr=0.0001 problem.hyp.alpha=0.01 problem/model=dt_net_recall_2d problem=mazes problem.hyp.lr_throttle=False problem.hyp.optimizer=sgd
