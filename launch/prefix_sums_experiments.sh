# These take ~2 hrs each on 1 gpu

#python train_model.py problem.hyp.alpha=1 problem.hyp.lr=0.0001 problem/model=dt_net_1d problem=prefix_sums name=prefix_sums_ablation
#python train_model.py problem.hyp.alpha=0 problem/model=dt_net_1d problem=prefix_sums name=prefix_sums_ablation
#python train_model.py problem.hyp.alpha=1 problem/model=dt_net_recall_1d problem=prefix_sums name=prefix_sums_ablation
#python train_model.py problem.hyp.alpha=0 problem/model=dt_net_recall_1d problem=prefix_sums name=prefix_sums_ablation
#python train_model.py problem.hyp.alpha=0 problem/model=ff_net_1d problem=prefix_sums name=prefix_sums_ablation
#python train_model.py problem.hyp.alpha=1 problem/model=ff_net_recall_1d problem=prefix_sums name=prefix_sums_ablation
#python train_model.py problem.hyp.alpha=0 problem/model=ff_net_recall_1d problem=prefix_sums name=prefix_sums_ablation

### TESTING

python test_model.py name=prefix_sums_ablation_test problem.model.test_iterations.low=201 problem.model.test_iterations.high=500 problem.model.model_path=../../prefix_sums_sanity3/training-phaseless-Shadai
python test_model.py name=prefix_sums_ablation_test problem.model.test_iterations.low=201 problem.model.test_iterations.high=500 problem.model.model_path=../../prefix_sums_sanity3/training-upcast-Rainer
python test_model.py name=prefix_sums_ablation_test problem.model.test_iterations.low=201 problem.model.test_iterations.high=500 problem.model.model_path=../../prefix_sums_sanity3/training-wettish-Corneilius
python test_model.py name=prefix_sums_ablation_test problem.model.test_iterations.low=201 problem.model.test_iterations.high=500 problem.model.model_path=../../prefix_sums_sanity3/training-released-Fabiana
python test_model.py name=prefix_sums_ablation_test problem.model.test_iterations.low=201 problem.model.test_iterations.high=500 problem.model.model_path=../../prefix_sums_sanity3/training-jussive-Rhet
python test_model.py name=prefix_sums_ablation_test problem.model.test_iterations.low=201 problem.model.test_iterations.high=500 problem.model.model_path=../../prefix_sums_sanity3/training-moony-Hilbert
python test_model.py name=prefix_sums_ablation_test problem.model.test_iterations.low=201 problem.model.test_iterations.high=500 problem.model.model_path=../../prefix_sums_sanity3/training-maungy-Chancie
python test_model.py name=prefix_sums_ablation_test problem.model.test_iterations.low=201 problem.model.test_iterations.high=500 problem.model.model_path=../../prefix_sums_sanity3/training-scandent-Elvis
python test_model.py name=prefix_sums_ablation_test problem.model.test_iterations.low=201 problem.model.test_iterations.high=500 problem.model.model_path=../../prefix_sums_sanity3/training-ablush-Thorn
python test_model.py name=prefix_sums_ablation_test problem.model.test_iterations.low=201 problem.model.test_iterations.high=500 problem.model.model_path=../../prefix_sums_sanity3/training-yeasty-Antonio
python test_model.py name=prefix_sums_ablation_test problem.model.test_iterations.low=201 problem.model.test_iterations.high=500 problem.model.model_path=../../prefix_sums_sanity3/training-ictic-Tawnia
python test_model.py name=prefix_sums_ablation_test problem.model.test_iterations.low=201 problem.model.test_iterations.high=500 problem.model.model_path=../../prefix_sums_sanity3/training-muscly-Verdell
python test_model.py name=prefix_sums_ablation_test problem.model.test_iterations.low=201 problem.model.test_iterations.high=500 problem.model.model_path=../../prefix_sums_sanity3/training-stirring-Aren
python test_model.py name=prefix_sums_ablation_test problem.model.test_iterations.low=201 problem.model.test_iterations.high=500 problem.model.model_path=../../prefix_sums_sanity3/training-candied-Kaleigh
python test_model.py name=prefix_sums_ablation_test problem.model.test_iterations.low=201 problem.model.test_iterations.high=500 problem.model.model_path=../../prefix_sums_sanity3/training-crabwise-Brittini
python test_model.py name=prefix_sums_ablation_test problem.model.test_iterations.low=201 problem.model.test_iterations.high=500 problem.model.model_path=../../prefix_sums_sanity3/training-awing-Rickita
python test_model.py name=prefix_sums_ablation_test problem.model.test_iterations.low=201 problem.model.test_iterations.high=500 problem.model.model_path=../../prefix_sums_sanity3/training-conchal-Selicia
python test_model.py name=prefix_sums_ablation_test problem.model.test_iterations.low=201 problem.model.test_iterations.high=500 problem.model.model_path=../../prefix_sums_sanity3/training-chilly-Alonzo
python test_model.py name=prefix_sums_ablation_test problem.model.test_iterations.low=201 problem.model.test_iterations.high=500 problem.model.model_path=../../prefix_sums_sanity3/training-flattest-Halee
