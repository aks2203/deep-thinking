python train_model.py problem.hyp.alpha=0 problem/model=ff_net_2d problem=chess name=chess_ablation
python train_model.py problem.hyp.alpha=0 problem/model=ff_net_recall_2d problem=chess name=chess_ablation
python train_model.py problem.hyp.alpha=0 problem/model=dt_net_2d problem=chess name=chess_ablation
python train_model.py problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=chess name=chess_ablation
python train_model.py problem.hyp.alpha=0.1 problem/model=dt_net_2d problem=chess name=chess_ablation
python train_model.py problem.hyp.alpha=0.1 problem/model=dt_net_recall_2d problem=chess name=chess_ablation
python train_model.py problem.hyp.alpha=0.5 problem/model=dt_net_2d problem=chess name=chess_ablation
python train_model.py problem.hyp.alpha=0.5 problem/model=dt_net_recall_2d problem=chess name=chess_ablation
