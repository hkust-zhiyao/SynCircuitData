import numpy as np
import argparse
from diffusion.utils import add_parent_path, set_seeds


add_parent_path(level=1)
from datasets.data import get_data, get_data_id, add_data_args


from experiment import GraphExperiment, add_exp_args


from model import get_model, get_model_id, add_model_args


from diffusion.optim.multistep import get_optim, get_optim_id, add_optim_args






parser = argparse.ArgumentParser()
add_data_args(parser)
add_exp_args(parser)
add_model_args(parser)
add_optim_args(parser)
args = parser.parse_args()
set_seeds(args.seed)





train_loader, eval_loader, test_loader, num_node_feat, num_node_classes, num_edge_classes, max_degree, augmented_feature_dict, initial_graph_sampler, eval_evaluator, test_evaluator, monitoring_statistics = get_data(args)

args.num_edge_classes = num_edge_classes
args.num_node_classes = num_node_classes

if args.final_prob_node is None:
    args.final_prob_node = [1-1e-12, 1e-12]
    args.num_node_classes = 2
    args.has_node_feature = False

if 0 in args.final_prob_edge:
    args.final_prob_edge[np.argmax(args.final_prob_edge)] = args.final_prob_edge[np.argmax(args.final_prob_edge)]-1e-12
    args.final_prob_edge[np.argmin(args.final_prob_edge)] = 1e-12

args.max_degree = max_degree
args.num_node_feat = num_node_feat
args.augmented_feature_dict = augmented_feature_dict



data_id = get_data_id(args)




model = get_model(args, initial_graph_sampler=initial_graph_sampler)
model_id = get_model_id(args)




optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)
optim_id = get_optim_id(args)




exp = GraphExperiment(args=args,
                 data_id=data_id,
                 model_id=model_id,
                 optim_id=optim_id,
                 train_loader=train_loader,
                 eval_loader=eval_loader,
                 test_loader=test_loader,
                 model=model,
                 optimizer=optimizer,
                 scheduler_iter=scheduler_iter,
                 scheduler_epoch=scheduler_epoch,
                 monitoring_statistics=monitoring_statistics,
                 eval_evaluator=eval_evaluator, 
                 test_evaluator=test_evaluator,
                 n_patient=50)

exp.run()
