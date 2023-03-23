export CUDA_VISIBLE_DEVICES=0,1; python train_model.py problem.hyp.alpha=0.01 problem/model=locrnn_ei_2d problem.model.width=128 problem=mazes name=mazes_locrnn_ei_prog_alpha_abl
export CUDA_VISIBLE_DEVICES=0,1; python train_model.py problem.hyp.alpha=0.05 problem/model=locrnn_ei_2d problem.model.width=128 problem=mazes name=mazes_locrnn_ei_prog_alpha_abl
export CUDA_VISIBLE_DEVICES=0,1; python train_model.py problem.hyp.alpha=0.001 problem/model=locrnn_ei_2d problem.model.width=128 problem=mazes name=mazes_locrnn_ei_prog_alpha_abl
export CUDA_VISIBLE_DEVICES=0,1; python train_model.py problem.hyp.alpha=0.15 problem/model=locrnn_ei_2d problem.model.width=128 problem=mazes name=mazes_locrnn_ei_prog_alpha_abl
export CUDA_VISIBLE_DEVICES=0,1; python train_model.py problem.hyp.alpha=0.1 problem/model=locrnn_ei_2d problem.model.width=128 problem=mazes name=mazes_locrnn_ei_prog_alpha_abl
