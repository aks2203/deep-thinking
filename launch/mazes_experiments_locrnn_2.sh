# python train_model.py problem.hyp.alpha=0.00 problem/model=locrnn_2d problem=mazes name=mazes_ablation
# python train_model.py problem.hyp.alpha=0.01 problem/model=locrnn_2d problem=mazes name=mazes_ablation
python train_model.py problem.hyp.alpha=0.50 problem/model=locrnn_2d problem=mazes name=mazes_ablation
python train_model.py problem.hyp.alpha=1.00 problem/model=locrnn_2d problem=mazes name=mazes_ablation
