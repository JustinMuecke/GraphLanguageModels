from functools import partial
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import json 
from tqdm import tqdm
import torch
from torch import nn
from typing import List, Tuple, Dict, Union, Optional, Generator
import random
import numpy as np
import sys
import os, sys
import logging
import wandb
import pandas as pd
import get_args
from models.graph_T5.classifier import GraphT5Classifier
from models.graph_T5.graph_t5 import T5TokenizerFast as T5Tokenizer
from models.graph_T5.wrapper_functions import Graph, graph_to_graphT5, graph_to_set_of_triplets, get_embedding, Data

PREFIXES : List[str] = ["AIO","EID","OIL","OILWI","OILWPI","UE","UEWI1","UEWI2","UEWPI","UEWIP","SOSINETO","CSC","OOR","OOD"]
MAX_LENGTH = 512

def load_data(kg, dataset_construction, radius, num_masked):
    #splits = ['train', 'dev', 'test']
    #fn_graphs = [Path(f"data/knowledgegraph/{kg}/relation_subgraphs_{dataset_construction}/num_neighbors=[1,2,2,2,2]/num_masked={num_masked}/radius={radius}/{split}_graphs.jsonl") for split in splits]
    #fn_labels = [Path(f"data/knowledgegraph/{kg}/relation_subgraphs_{dataset_construction}/num_neighbors=[1,2,2,2,2]/num_masked={num_masked}/radius={radius}/{split}_labels.jsonl") for split in splits]
    #fn_label2index = Path(f"data/knowledgegraph/{kg}/relation_subgraphs_{dataset_construction}/num_neighbors=[1,2,2,2,2]/label2index.json")

    df = pd.read_csv("dataset.csv", header=0, nrows=10)
    subset_df = df[df["tokenized_length"] < MAX_LENGTH]
    df["labels"] = df["file_name"].map(lambda x : "Inconsistent" if x.split("_")[0] in PREFIXES else "Consistent")
    data = df["body"].map(lambda x : x.replace(" ", "").replace("\'", "\"")).tolist()
 
    all_labels = df["labels"].tolist()
    labels = []
    graphs = []
    for triples, label in tqdm(zip(data, all_labels)):
        jsonified = json.loads(triples)
        while jsonified:
            try:
                graphs.append(Graph(jsonified))
                labels.append(label)
                break
            except: 
                print("Sh")
                jsonified = jsonified[:-1]
                continue 
    #graphs = [Graph(json.loads(triples)) for triples in data]

    #
    #graphs = {split: [Graph(json.loads(l)) for l in tqdm(fn.open('r'))] for split, fn in zip(splits, fn_graphs)}
#
#    labels = {split: fn.open('r').readlines() for split, fn in zip(splits, fn_labels)}
#    for split in splits:
#        labels[split] = [l.strip() for l in labels[split] if l.strip()]
#
    label_to_index = {"Inconsistent" : 1, "Consistent" : 0}
##  
    graph_dict = {"train": graphs[:int(0.7*len(graphs))], "test": graphs[int(0.7*len(graphs)):]}
    label_dict = {"train" : labels[:int(0.7*len(labels))], "test": labels[int(0.7*len((labels))):]}

#    assert set(labels['train']) == set(labels['dev']) == set(labels['test']), (set(labels['train']), set(labels['dev']), set(labels['test']))
    return graph_dict, label_dict, label_to_index

def data_to_dataT5(graph:Graph, tokenizer:T5Tokenizer, label:str, label_to_index:dict, graph_representation:str, eos:str):
    """
    :param graph: graph to convert
    :param tokenizer: tokenizer of model
    :param label: label of the relation
    :param label_to_index: mapping from label to index
    :param graph_representation: how to represent the graph. 
    :param eos: end-of-sequence token. Can be `False` for not using an eos token. When using an eos token, there are two ways to use it: `bidirectional` means that the eos token is connected to every other node in the graph, with a relative position of positive infinity (from node to eos) or negative infinity (from eos to node). `unidirectional` means that the eos token is connected to every node in the graph with a relative position of positive infinity (from node to eos), but not the other way around (i.e. no connection from eos to other node). This means, that nodes do not get messages from the eos token, which preserves locality when using the local GLM
    """
    if graph_representation == 'lGLM':
        data = graph_to_graphT5(graph, tokenizer, how='local', eos=eos)
    elif graph_representation == 'set':
        data = graph_to_set_of_triplets(graph, tokenizer, order='random')
    elif graph_representation == 'gGLM':
        data = graph_to_graphT5(graph, tokenizer, how='global', eos=eos)
    elif graph_representation == 'list':
        data = graph_to_set_of_triplets(graph, tokenizer, order='alphabetical')
    else:
        raise ValueError(f"unknown graph_representation {graph_representation}")
    data.label = torch.tensor(label_to_index[label])
    return data

def get_batch(data_instances:List[Data], pad_token_id:int, device:str):
    """
    can be implemented more efficiently with nested tensors, but they are currently unstable
    """
    max_seq_len = max([data.input_ids.shape[1] for data in data_instances])

    if data_instances[0].relative_position is None:
        assert data_instances[0].sparsity_mask is None
        assert data_instances[0].use_additional_bucket is None
        is_sequence_transformer = True
    else:
        assert data_instances[0].sparsity_mask is not None
        assert data_instances[0].use_additional_bucket is not None
        is_sequence_transformer = False

    # intialize tensors
    input_ids = torch.ones((len(data_instances), max_seq_len), dtype=torch.long, device=device) * pad_token_id
    if not is_sequence_transformer:
        relative_position = torch.zeros((len(data_instances), max_seq_len, max_seq_len), dtype=torch.long, device=device)
        sparsity_mask = torch.zeros((len(data_instances), max_seq_len, max_seq_len), dtype=torch.bool, device=device)
        use_additional_bucket = torch.zeros((len(data_instances), max_seq_len, max_seq_len), dtype=torch.bool, device=device)

    # fill tensors
    for i, data in enumerate(data_instances):
        input_ids[i, :data.input_ids.shape[1]] = data.input_ids
        if not is_sequence_transformer:
            relative_position[i, :data.relative_position.shape[1], :data.relative_position.shape[2]] = data.relative_position
            sparsity_mask[i, :data.sparsity_mask.shape[1], :data.sparsity_mask.shape[2]] = data.sparsity_mask
            use_additional_bucket[i, :data.use_additional_bucket.shape[1], :data.use_additional_bucket.shape[2]] = data.use_additional_bucket

    if is_sequence_transformer:
        relative_position = None # [None] * len(data_instances)
        sparsity_mask = None # [None] * len(data_instances)
        use_additional_bucket = None # [None] * len(data_instances)

    indices = [data.indices for data in data_instances]
    label = torch.tensor([data.label for data in data_instances], device=device)

    return input_ids, relative_position, sparsity_mask, use_additional_bucket, indices, label

def chunker(data_list:List[Data], batch_size:int):
    """
    returns a generator that yields batches of size batch_size
    """
    return (data_list[pos:pos + batch_size] for pos in range(0, len(data_list), batch_size))

def get_accuracy(preds:torch.Tensor, label:torch.Tensor):
    """
    :param preds: shape (batch_size, num_classes)
    :param label: shape (batch_size)
    """
    return (preds.argmax(dim=1) == label).sum().item() / len(label) * 100

def run_eval_epoch(model:GraphT5Classifier, data:List[Data], criterion:nn.Module, batch_size:int, device:str):
    with torch.no_grad():
        losses = []
        accuracies = []
        weights = []

        for data_instances in chunker(data, batch_size):
            # create batch
            logging.debug("get batch")
            input_ids, relative_position, sparsity_mask, use_additional_bucket, indices, label = get_batch(data_instances, pad_token_id=model.tokenizer.pad_token_id, device=device)

            logging.debug("forward")
            logits = model.forward(
                input_ids=input_ids,
                relative_position=relative_position,
                sparsity_mask=sparsity_mask,
                use_additional_bucket=use_additional_bucket,
            )

            logging.debug("get embedding")
            logits = torch.cat([
                get_embedding(sequence_embedding=logits[i], indices=indices[i], concept='<mask>', embedding_aggregation='mean')
                for i in range(len(data_instances))
            ], dim=0)

            logging.debug("get loss and accuracy")
            loss = criterion(logits, label)
            accuracy = get_accuracy(logits, label)

            losses.append(loss.item())
            accuracies.append(accuracy)
            weights.append(len(label)) 
        
        logging.debug("aggregate loss and accuracy")
        loss = np.average(losses, weights=weights)
        accuracy = np.average(accuracies, weights=weights)
    return loss, accuracy

def run_train_epoch(model:GraphT5Classifier, data:List[Data], criterion:nn.Module, optimizer:torch.optim.Optimizer, batch_size:int, gradient_accumulation_steps:int, device:str):
    losses = []
    accuracies = []
    weights = []
    optimizer.zero_grad()

    random.shuffle(data)

    for i, data_instances in tqdm(enumerate(chunker(data, batch_size)), total=len(data)//batch_size):
        # create batch
        input_ids, relative_position, sparsity_mask, use_additional_bucket, indices, label = get_batch(data_instances, pad_token_id=model.tokenizer.pad_token_id, device=device)

        logits = model.forward(
            input_ids=input_ids,
            relative_position=relative_position,
            sparsity_mask=sparsity_mask,
            use_additional_bucket=use_additional_bucket,
        )

        logits = torch.cat([
            get_embedding(sequence_embedding=logits[i], indices=indices[i], concept='<mask>', embedding_aggregation='mean')
            for i in range(len(data_instances))
        ], dim=0)

        loss = criterion(logits, label)
        # loss = logits.sum()

        loss.backward()

        if (i+1) % gradient_accumulation_steps == 0 or (i+1) == len(data)//batch_size:
            optimizer.step()
            optimizer.zero_grad()

        accuracy = get_accuracy(logits, label)
        losses.append(loss.item())
        accuracies.append(accuracy)
        weights.append(len(label)) 

    loss = np.average(losses, weights=weights)
    accuracy = np.average(accuracies, weights=weights)
    return loss, accuracy

def main(args):
    if not args.device.startswith('cuda'):
        logging.warning(f'using CPU {args.device}, training might be slow.')
    else:
        logging.info(f'using GPU {args.device}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    logging.info('load data')
    graphs, labels, label_to_index = load_data(kg=args.kg, dataset_construction=args.dataset_construction, radius=args.radius, num_masked=args.num_masked)
    
    logging.info('load T5 encoder')
    num_classes = len(label_to_index)
    model = GraphT5Classifier(config=GraphT5Classifier.get_config(num_classes=num_classes, modelsize=args.modelsize, num_additional_buckets=args.num_additional_buckets))
    
    if args.num_additional_buckets != 0:
        logging.info(f'init relative position bias with {args.num_additional_buckets} additional buckets')
        model.t5model.init_relative_position_bias(modelsize=args.modelsize, init_decoder=False, init_additional_buckets_from=args.init_additional_buckets_from)

    if args.reset_params:
        logging.info('resetting model parameters')
        get_args.reset_params(model=model)
    #model.to(args.device)

    if not args.reload_data:
        logging.info('convert data to T5 input')
        data = {split: [data_to_dataT5(graph, model.tokenizer, label, label_to_index, args.graph_representation, eos=args.eos_usage) for graph, label in tqdm(zip(graphs[split], labels[split]), total=len(labels[split]))] for split in ['train', 'test']}

    # loss and optimizer
    criterion = args.criterion()

    # params_to_train = str2params_to_train(s=args.params_to_train, model=model)
    # freeze_params(s=args.params_to_train, model=model)
    # optimizer = args.optimizer(params_to_train, lr=args.learning_rate)
    optimizer = args.optimizer(model.parameters(), lr=args.learning_rate)

    best_epoch = 0
    best_dev_accuracy = 0
    best_dev_loss = float('inf')
    best_test_accuracy = 0
    best_test_loss = float('inf')
    stopped_early = False

    "Training "
    logging.info('train the model')
    for epoch in range(args.num_epochs):
        if args.reload_data:
            logging.info('convert data to T5 input')
            data = {split: [data_to_dataT5(graph, model.tokenizer, label, label_to_index, args.graph_representation, eos=args.eos_usage) for graph, label in tqdm(zip(graphs[split], labels[split]), total=len(labels[split]))] for split in ['train', 'test']}
            logging.info('train epoch')
        train_loss, train_accuracy = run_train_epoch(model=model, data=data['train'], criterion=criterion, optimizer=optimizer, batch_size=args.train_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, device=args.device)
        logging.info(f'train - {epoch = } # {train_loss = :.2f} # {train_accuracy = :.2f}')

        # get dev scores
 #       dev_loss, dev_accuracy = run_eval_epoch(model=model, data=data['dev'], criterion=criterion, batch_size=args.eval_batch_size, device=args.device)
 #       logging.info(f'dev   - {epoch = } # {dev_loss = :.2f} # {dev_accuracy = :.2f}')

        # get test scores
        test_loss, test_accuracy = run_eval_epoch(model=model, data=data['test'], criterion=criterion, batch_size=args.eval_batch_size, device=args.device)
        logging.info(f'test  - {epoch = } # {test_loss = :.2f} # {test_accuracy = :.2f}')

        if train_loss < best_dev_loss:
            best_epoch = epoch
            best_dev_accuracy = train_accuracy
            best_dev_loss = train_loss
            best_test_accuracy = train_accuracy
            best_test_loss = train_loss

        wandb.log(
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "stopped_early": float(stopped_early),
                "train/accuracy": train_accuracy, "train/loss": train_loss, 
 #               "dev/accuracy": dev_accuracy, "dev/loss": dev_loss, 'dev/best_accuracy': best_dev_accuracy, 'dev/best_loss': best_dev_loss,
                "test/accuracy": test_accuracy, "test/loss": test_loss, 'test/best_accuracy': best_test_accuracy, 'test/best_loss': best_test_loss,
            }
        )

        last_epoch = epoch
        if epoch - best_epoch >= args.early_stopping:
            logging.info(f'stopped early at epoch {epoch}')
            stopped_early = True
            break

    for epoch in range(last_epoch+1, args.num_epochs):
        wandb.log(
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "stopped_early": float(stopped_early),
                "train/accuracy": train_accuracy, "train/loss": train_loss, 
 #               "dev/accuracy": dev_accuracy, "dev/loss": dev_loss, 'dev/best_accuracy': best_dev_accuracy, 'dev/best_loss': best_dev_loss,
                "test/accuracy": test_accuracy, "test/loss": test_loss, 'test/best_accuracy': best_test_accuracy, 'test/best_loss': best_test_loss,
            }
        )


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter  # makes wandb log the default values
    )
    get_args.add_args_shared(parser)
    get_args.add_args(parser)
    args = get_args.load_args(parser)

    args.device = 'cuda' if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    
    # logging
    root = logging.getLogger()
    root.setLevel(args.logging_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(args.logging_level)
    formatter = logging.Formatter(f"%(asctime)s [%(levelname)s] %(filename)s, Line %(lineno)d\n%(message)s",datefmt=f"%H:%M:%S",)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # logging.basicConfig(
    #     level=args.logging_level,
    #     # format=f"%(asctime)s [%(levelname)s] %(message)s (Line %(lineno)d in %(filename)s)",
    #     format=f"%(asctime)s [%(levelname)s] %(filename)s, Line %(lineno)d\n%(message)s",
    #     datefmt=f"%H:%M:%S",
    # )

    # wandb
    name = f'GLM_TEST_LOADING'
    wandb_run = wandb.init(
        mode=args.wandb_mode,
        project="GLM-link_prediction-long_train",
        name=name,
        # Track hyperparameters and run metadata
        config=args.__dict__,
        group=f'{name}_lr={args.learning_rate}_resetparams={args.reset_params}_modelsize={args.modelsize}_eos={args.eos_usage}',
        tags=['LM']
    )

    main(args)

    logging.info("done with main")
