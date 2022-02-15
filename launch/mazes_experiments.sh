python train_model.py problem.hyp.alpha=0.00 problem/model=dt_net_2d problem=mazes name=mazes_ablation
python train_model.py problem.hyp.alpha=0.01 problem/model=dt_net_2d problem=mazes name=mazes_ablation
python train_model.py problem.hyp.alpha=0.50 problem/model=dt_net_2d problem=mazes name=mazes_ablation
python train_model.py problem.hyp.alpha=1.00 problem/model=dt_net_2d problem=mazes name=mazes_ablation
python train_model.py problem.hyp.alpha=0.00 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation
python train_model.py problem.hyp.alpha=0.01 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation
python train_model.py problem.hyp.alpha=0.50 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation
python train_model.py problem.hyp.alpha=1.00 problem/model=dt_net_recall_2d problem=mazes name=mazes_ablation
python train_model.py problem.hyp.alpha=0.00 problem.hyp.epochs=200 problem.hyp.lr=0.0001 problem.hyp.lr_schedule=[175] problem/model=ff_net_2d problem=mazes name=mazes_ablation
python train_model.py problem.hyp.alpha=0.01 problem.hyp.epochs=200 problem.hyp.lr=0.0001 problem.hyp.lr_schedule=[175] problem/model=ff_net_2d problem=mazes name=mazes_ablation
python train_model.py problem.hyp.alpha=0.50 problem.hyp.epochs=200 problem.hyp.lr=0.0001 problem.hyp.lr_schedule=[175] problem/model=ff_net_2d problem=mazes name=mazes_ablation
python train_model.py problem.hyp.alpha=0.00 problem.hyp.epochs=200 problem.hyp.lr=0.0001 problem.hyp.lr_schedule=[175] problem/model=ff_net_recall_2d problem=mazes name=mazes_ablation
python train_model.py problem.hyp.alpha=0.01 problem.hyp.epochs=200 problem.hyp.lr=0.0001 problem.hyp.lr_schedule=[175] problem/model=ff_net_recall_2d problem=mazes name=mazes_ablation
python train_model.py problem.hyp.alpha=0.50 problem.hyp.epochs=200 problem.hyp.lr=0.0001 problem.hyp.lr_schedule=[175] problem/model=ff_net_recall_2d problem=mazes name=mazes_ablation
