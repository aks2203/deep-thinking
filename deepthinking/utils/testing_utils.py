""" testing_utils.py
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


def test(net, loaders, mode, iters, problem, device, disable_tqdm=False):
    accs = []
    for loader in loaders:
        if mode == "default":
            accuracy = test_default(net, loader, iters, problem, device, disable_tqdm)
        elif mode == "max_conf":
            accuracy = test_max_conf(net, loader, iters, problem, device, disable_tqdm)
        elif mode == "default_huge_dataset":
            accuracy = test_default_huge_dataset(net, loader, iters, problem, device, disable_tqdm)
        elif mode == "default_zero":
            accuracy = test_default_zero(net, loader, iters, problem, device, disable_tqdm)
        elif mode == "default_small_edit":
            accuracy = test_default_small_edit(net, loader, iters, problem, device, disable_tqdm)
        elif mode == "default_sudoku":
            accuracy = test_default_sudoku(net, loader, iters, problem, device, disable_tqdm)
        elif mode == "max_conf_and_default":
            accuracy = test_max_conf_and_default(net, loader, iters, problem, device, disable_tqdm)
        else:
            raise ValueError(f"{ic.format()}: test_{mode}() not implemented.")
        accs.append(accuracy)
    return accs


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
    elif problem in ["sudoku_5", "sudoku_4", "sudoku_3", "sudoku_2"]:
        predicted = predicted * (inputs.max(1)[0].view(inputs.size(0), -1) == 0)
        predicted = predicted + inputs.max(1)[0].view(inputs.size(0), -1)

    return predicted


def isinRange(board, input, to_print):
    N = 9
    for i in range(0, N):
        for j in range(0, N):
            if ((board[i][j] <= 0) or
                    (board[i][j] > 9)):
                return False
            if input[i][j] != 0:
                if input[i][j] != board[i][j]:
                    if to_print:
                        print(i,j)
                    return False

    return True


def isValidSudoku(board, input, to_print):
    N = 9
    if (isinRange(board, input, to_print) == False):
        return False
    unique = [False] * (N + 1)

    for i in range(0, N):

        for m in range(0, N + 1):
            unique[m] = False

        for j in range(0, N):
            Z = int(board[i][j])
            if (unique[Z] == True):
                return False
            unique[Z] = True

    for i in range(0, N):
        for m in range(0, N + 1):
            unique[m] = False
        for j in range(0, N):
            Z = int(board[j][i])
            if (unique[Z] == True):
                return False
            unique[Z] = True

    for i in range(0, N - 2, 3):
        for j in range(0, N - 2, 3):
            for m in range(0, N + 1):
                unique[m] = False
            for k in range(0, 3):
                for l in range(0, 3):
                    X = i + k
                    Y = j + l
                    Z = int(board[X][Y])
                    if (unique[Z] == True):
                        return False
                    unique[Z] = True
    return True


def find_sudoku_correct(inputs, predicted, to_print):
    corrects = 0
    for bs in range(inputs.shape[0]):
        input = inputs[bs]
        predict = predicted[bs]
        if(isValidSudoku(predict, input, to_print)):
            corrects+=1
            # if to_print:
            #     print("Valid")

    if to_print:
        print(corrects)

    return corrects


def test_default_sudoku(net, testloader, iters, problem, device, disable_tqdm):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters)
    total = 0
    batch = 0
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False, disable=disable_tqdm):
            inputs, targets = inputs.to(device), targets.to(device)

            all_outputs = net(inputs, iters_to_do=max_iters)

            for i in range(all_outputs.size(1)):
                if i % 100 == 0:
                    print(i)
                outputs = all_outputs[:, i]
                outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
                predicted = get_predicted(inputs, outputs, problem)
                targets = targets.view(targets.size(0), -1)
                if i+1 in iters:
                    predicted = predicted.reshape(predicted.size(0), 9, 9)
                    inputs_r = inputs.reshape(inputs.size(0), 9, 9)
                    to_print = False
                    if i == max_iters - 1:
                        to_print = True
                    corrects[i] += find_sudoku_correct(inputs_r, predicted, to_print)

            total += targets.size(0)
            batch += 1
            # if batch == 5:
            #     break

    accuracy = 100.0 * corrects / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite - 1].item()
    return ret_acc


def test_default(net, testloader, iters, problem, device, disable_tqdm):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters)
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False, disable=disable_tqdm):
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

def test_default_huge_dataset(net, testloader, iters, problem, device, disable_tqdm):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(1)
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False, disable=disable_tqdm):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs, iters_to_do=max_iters)
            predicted = get_predicted(inputs, outputs, problem)
            targets = targets.view(targets.size(0), -1)
            corrects += torch.amin(predicted == targets, dim=[1]).sum().item()

            total += targets.size(0)
            print(corrects)

    accuracy = 100.0 * corrects / total
    return accuracy


def test_default_zero(net, testloader, iters, problem, device, disable_tqdm):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters)
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False, disable=disable_tqdm):
            inputs, targets = inputs.to(device), targets.to(device)

            net.train()
            outputs_before, interim_thought = net(inputs, iters_to_do=1)
            interim_thought = interim_thought.detach()
            interim_thought = torch.zeros_like(interim_thought)

            net.eval()

            all_outputs = net(inputs, iters_to_do=max_iters, interim_thought=interim_thought)

            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]
                predicted = get_predicted(inputs, outputs, problem)
                targets = targets.view(targets.size(0), -1)
                corrects[i] += torch.amin(predicted == targets, dim=[1]).sum().item()

            total += targets.size(0)

            break

    accuracy = 100.0 * corrects / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite - 1].item()
    return ret_acc




def test_max_conf(net, testloader, iters, problem, device, disable_tqdm):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters).to(device)
    total = 0
    softmax = torch.nn.functional.softmax

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False, disable=disable_tqdm):
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



def green(inputs, row, col):
    inputs[0, row:row + 2, col:col + 2] = 0
    inputs[1, row:row + 2, col:col + 2] = 1
    inputs[2, row:row + 2, col:col + 2] = 0

    return inputs


def red(inputs, row, col):
    inputs[0, row:row + 2, col:col + 2] = 1
    inputs[1, row:row + 2, col:col + 2] = 0
    inputs[2, row:row + 2, col:col + 2] = 0

    return inputs


def white(inputs, row, col):
    inputs[0, row:row + 2, col:col + 2] = 1
    inputs[1, row:row + 2, col:col + 2] = 1
    inputs[2, row:row + 2, col:col + 2] = 1

    return inputs

def mod(inputs, targets):
    length = int(targets.sum()/4)
    if length < 3:
        return inputs, mod


    for i in range(inputs.size(1)):
        for j in range(inputs.size(2)):
            if inputs[0, i, j] == 1 and inputs[1, i, j] == 0 and inputs[2, i, j] == 0:
                #print("red", i, j)
                # check the path
                #left
                if targets[i, j-1] == 1:
                    #print("left", inputs[:, i, j-1])
                    inputs = red(inputs, i, j-4)
                    inputs = white(inputs, i, j)

                    targets[i:i + 2, j:j + 2] = 0
                    targets[i:i+2, j-2:j] = 0
                    return inputs, targets

                #right
                if targets[i, j+2] == 1:
                    #print("right", inputs[:, i, j+2])
                    inputs = red(inputs, i, j + 4)
                    inputs = white(inputs, i, j)

                    targets[i:i + 2, j:j + 2] = 0
                    targets[i:i + 2, j+2:j + 4] = 0
                    return inputs, targets

                # top
                if targets[i-1, j] == 1:
                    #print("top", inputs[:, i-1, j])
                    inputs = red(inputs, i-4, j)
                    inputs = white(inputs, i, j)

                    targets[i:i + 2, j:j + 2] = 0
                    targets[i - 2:i, j:j + 2] = 0
                    return inputs, targets

                # bottom
                if targets[i + 2, j] == 1:
                    #print("bottom", inputs[:, i + 2, j])
                    inputs = red(inputs, i+4, j)
                    inputs = white(inputs, i, j)

                    targets[i:i + 2, j:j + 2] = 0
                    targets[i+2:i + 4, j:j + 2] = 0
                    return inputs, targets


def get_small_mod(inputs, targets):
    inputs_mod = torch.clone(inputs)
    targets_mod = torch.clone(targets)
    for i in range(inputs.size(0)):
        inputs_mod[i, :, :, :], targets_mod[i, :, :] = mod(inputs_mod[i, :, :, :], targets_mod[i, :, :])

    return inputs_mod, targets_mod


def test_default_small_edit(net, testloader, iters, problem, device, disable_tqdm):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters)
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False, disable=disable_tqdm):
            inputs, targets = inputs.to(device), targets.to(device)

            net.train()
            outputs_before, interim_thought = net(inputs, iters_to_do=max_iters)
            interim_thought = interim_thought.detach()


            inputs, targets = get_small_mod(inputs, targets)

            net.eval()

            all_outputs = net(inputs, iters_to_do=max_iters, interim_thought=interim_thought)

            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]
                predicted = get_predicted(inputs, outputs, problem)
                targets = targets.view(targets.size(0), -1)
                corrects[i] += torch.amin(predicted == targets, dim=[1]).sum().item()

            total += targets.size(0)

            break

    accuracy = 100.0 * corrects / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite - 1].item()
    return ret_acc


def test_max_conf_and_default(net, testloader, iters, problem, device, disable_tqdm):
    print(iters)
    max_iters = max(iters)
    net.eval()
    corrects_max_conf_all = torch.zeros(max_iters).to(device)
    corrects_default_all  = torch.zeros(max_iters).to(device)
    wrong_pixels_all = torch.zeros(max_iters).to(device)

    total = 0
    softmax = torch.nn.functional.softmax

    with torch.no_grad():
        index = 0
        for inputs, targets in tqdm(testloader, leave=False, disable=disable_tqdm):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(targets.size(0), -1)
            total += targets.size(0)

            corrects_max_conf, corrects_default, wrong_pixels = net(inputs, iters_to_do=max_iters, targets=targets, problem=problem)
            corrects_max_conf_all += corrects_max_conf
            corrects_default_all += corrects_default
            wrong_pixels_all += wrong_pixels
            index+=1
            if index == 10:
                break

    accuracy_max_conf = 100 * corrects_max_conf_all.long().cpu() / total
    accuracy_default_all = 100 * corrects_max_conf_all.long().cpu() / total
    wrong_pixels_all = wrong_pixels_all.cpu()/ total

    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = (accuracy_max_conf[ite-1].item(), accuracy_default_all[ite-1].item(), wrong_pixels_all[ite-1].item())
    return ret_acc