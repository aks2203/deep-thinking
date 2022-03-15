""" testing.py
    Utilities for testing models

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import einops
import torch
from icecream import ic
from tqdm import tqdm

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115, C0114).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, C0114


def test(net, loaders, mode, iters, problem, device):
    accs = []
    for loader in loaders:
        if mode == "default":
            accuracy = test_default(net, loader, iters, problem, device)
        elif mode == "max_conf":
            accuracy = test_max_conf(net, loader, iters, problem, device)
        elif mode == "convergence":
            accuracy = test_convergence(net, loader, iters, problem, device)
        else:
            raise ValueError(f"{ic.format()}: test_{mode}() not implemented.")
        accs.append(accuracy)
    return accs


def get_confidence(inputs, outputs, problem):
    outputs = torch.nn.Softmax(dim=1)(outputs.clone())
    confidence = outputs[:, 1]
    confidence = confidence.view(confidence.size(0), -1)
    if problem == "mazes":
        confidence = confidence * (inputs.max(1)[0].view(inputs.size(0), -1))
    return confidence


def get_predicted(inputs, outputs, problem):
    outputs = outputs.clone()
    predicted = outputs.argmax(1)
    predicted = predicted.view(predicted.size(0), -1)
    if problem == "mazes":
        predicted = predicted * (inputs.max(1)[0].view(inputs.size(0), -1))
    elif problem == "chess":
        outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
        top_2 = torch.topk(outputs[:, 1], 2, dim=1)[0].min(dim=1)[0]
        top_2 = einops.repeat(top_2, "n -> n k", k=8)
        top_2 = einops.repeat(top_2, "n m -> n m k", k=8).view(-1, 64)
        outputs[:, 1][outputs[:, 1] < top_2] = -float("Inf")
        outputs[:, 0] = -float("Inf")
        predicted = outputs.argmax(1)

    return predicted


def test_default(net, testloader, iters, problem, device):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters)
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            all_outputs = net(inputs, iters_to_do=max_iters)

            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]
                predicted = get_predicted(inputs, outputs, problem)
                targets = targets.view(targets.size(0), -1)
                corrects[i] += torch.amin(predicted == targets, dim=[1]).sum().item()

            total += targets.size(0)

    accuracy = 100.0 * corrects / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
    return ret_acc


def test_max_conf(net, testloader, iters, problem, device):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters).to(device)
    total = 0
    softmax = torch.nn.functional.softmax

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(targets.size(0), -1)
            total += targets.size(0)


            all_outputs = net(inputs, iters_to_do=max_iters)

            confidence_array = torch.zeros(max_iters, inputs.size(0)).to(device)
            corrects_array = torch.zeros(max_iters, inputs.size(0)).to(device)
            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]
                conf = softmax(outputs.detach(), dim=1).max(1)[0]
                conf = conf.view(conf.size(0), -1)
                if problem == "mazes":
                    conf = conf * inputs.max(1)[0].view(conf.size(0), -1)
                confidence_array[i] = conf.sum([1])
                predicted = get_predicted(inputs, outputs, problem)
                corrects_array[i] = torch.amin(predicted == targets, dim=[1])

            correct_this_iter = corrects_array[torch.cummax(confidence_array, dim=0)[1],
                                               torch.arange(corrects_array.size(1))]
            corrects += correct_this_iter.sum(dim=1)

    accuracy = 100 * corrects.long().cpu() / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
    return ret_acc


def test_convergence(net, testloader, iters, problem, device):
    max_iters = max(iters)
    max_iters_used = 0
    net.train()
    corrects = torch.zeros(1)
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            old_outputs, interim_thought = net(inputs, iters_to_do=1)
            old_confidence = get_confidence(inputs, old_outputs, problem)
            done = False
            ite = 1
            while ite < max_iters and not done:
                new_outputs, interim_thought = net(inputs,
                                                   iters_elapsed=ite,
                                                   iters_to_do=1,
                                                   interim_thought=interim_thought)

                #     This threshold (1e-3) could be tuned, or perhaps set differently for
                #     each problem. For now, this is a reasonable choice for the three problems
                #     we have implmented (-Avi, Mar 15, 2022)
                # for each input, decide whether to stop
                new_confidence = get_confidence(inputs, new_outputs, problem)
                stop_here = torch.mean(torch.abs(new_confidence - old_confidence), dim=1) <= 1e-3

                # count accuracy on inputs that stop here
                if torch.any(stop_here):
                    predicted_here = get_predicted(inputs[stop_here], new_outputs[stop_here], problem)
                    targets_here = targets[stop_here].view(targets[stop_here].size(0), -1)
                    corrects[0] += torch.amin(predicted_here == targets_here, dim=[1]).sum().item()

                # update running variables
                if torch.any(~stop_here):
                    inputs = inputs[~stop_here]
                    targets = targets[~stop_here]
                    interim_thought = interim_thought[~stop_here]
                    old_confidence = new_confidence[~stop_here]
                else:
                    done = True
                ite += 1
            if ite > max_iters_used:
                max_iters_used = ite

    accuracy = 100.0 * corrects / total
    ret_acc = {max_iters_used: accuracy.item()}
    return ret_acc




