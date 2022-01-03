python train_model.py problem.hyp.alpha=1 problem/model=dt_net_1d problem=prefix_sums name=prefix_sums_ablation
python train_model.py problem.hyp.alpha=0 problem/model=dt_net_1d problem=prefix_sums name=prefix_sums_ablation
python train_model.py problem.hyp.alpha=1 problem/model=dt_net_recall_1d problem=prefix_sums name=prefix_sums_ablation
python train_model.py problem.hyp.alpha=0 problem/model=dt_net_recall_1d problem=prefix_sums name=prefix_sums_ablation
python train_model.py problem.hyp.alpha=0 problem/model=ff_net_1d problem=prefix_sums name=prefix_sums_ablation
python train_model.py problem.hyp.alpha=1 problem/model=ff_net_recall_1d problem=prefix_sums name=prefix_sums_ablation
python train_model.py problem.hyp.alpha=0 problem/model=ff_net_recall_1d problem=prefix_sums name=prefix_sums_ablation

