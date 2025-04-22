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
from transformers import LongT5ForConditionalGeneration, AutoTokenizer
import ast
from models.graph_T5.wrapper_functions import Graph
from torch.optim import AdamW
from models.T5WithClassificationHead import T5WithClassificationHead
from sklearn.metrics import precision_score, recall_score
from datetime import datetime
def load_data():
    train_df = pd.read_csv("../data/train_data.csv", header=0)
    eval_df = pd.read_csv("../data/eval_data.csv", header=0)
    test_df = pd.read_csv("../data/test_data.csv", header=0)

    train_graphs = [Graph(ast.literal_eval(triples)) for triples in train_df["body"]]
    eval_graphs = [Graph(ast.literal_eval(triples))for triples in eval_df["body"]]
    test_graphs = [Graph(ast.literal_eval(triples))for triples in test_df["body"]]

    graph_dict= {"train" : train_graphs, "eval":eval_graphs, "test":test_graphs}

    label_2_index = {"Inconsistent" : 1, "Consistent" : 0}
    label_dict= {"train" : train_df["consistency"].map(label_2_index), "eval" : eval_df["consistency"].map(label_2_index), "test" : test_df["consistency"].map(label_2_index)}
    return graph_dict, label_dict, label_2_index

def graph_to_t5_input(graph: Graph) -> str:
    triples = graph.g
    return " ".join([f"{s} {p} {o}." for s, p, o in triples])

def graphs_to_t5_format(graphs: List[Graph], labels: List[str]) -> List[Dict[str, str]]:
    return [
        {
            "input_ids": graph_to_t5_input(graph),
            "label": label
        }
        for graph, label in tqdm(zip(graphs, labels), total=len(graphs))
    ]

def tokenize_data(tokenizer, data, max_input_length=4096, max_target_length=10):
    task_prefix = "check for consistency: "
    
    input_texts = [task_prefix + d["input_ids"] for d in data]
    target_labels = [d["label"] for d in data]

    # Initialize lists to store the tokenized sequences
    model_inputs = {
        "input_ids": [],
        "attention_mask": []
    }

    # Tokenize the input texts individually and extract the needed attributes
    for text in tqdm(input_texts, total=len(input_texts)):
        encoding = tokenizer(
            text,
            padding="max_length",  # Ensure all sequences are padded to max_input_length
            max_length=max_input_length,
            truncation=True,
            return_tensors="pt"
        )
        model_inputs["input_ids"].append(encoding.input_ids)
        model_inputs["attention_mask"].append(encoding.attention_mask)

    labels = torch.tensor(target_labels, dtype=torch.long)
    model_inputs["labels"] = labels
    return model_inputs


def run_train_epoch(model, data, criterion, optimizer, batch_size, gradient_accumulation_steps, device):
    model.train()
    total_loss = 0
    total_accuracy = 0
    batch_size = 1
#    data_points=10
    data_points = int(len(data["input_ids"]))
    step = 0
    # Loop through data with batch size of 1
    for i in tqdm(range(0, data_points, batch_size)):
        step += 1
    # Get the batch data

        input_ids = torch.stack(data["input_ids"][i:i + batch_size]).squeeze(1).to(device)
        attention_mask = torch.stack(data["attention_mask"][i:i + batch_size]).squeeze(1).to(device)
        labels = torch.Tensor(data["labels"][i:i + batch_size]).to(device)

        # If needed, stack into batch tensors (if the slices are not already batched)
       # if isinstance(input_ids, list):
       #     input_ids = torch.stack(input_ids)
       #     attention_mask = torch.stack(attention_mask)
       #     labels = torch.stack(labels)



        loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()

        # Compute accuracy
        preds = (torch.sigmoid(logits) > 0.5).long()
        accuracy = (preds == labels).float().mean().item()
        print(logits)
        print(labels)
        print(preds)
        total_loss += loss.item()
        total_accuracy += accuracy

    # Averages
    avg_loss = total_loss / step
    avg_accuracy = total_accuracy / step

    return avg_loss, avg_accuracy

def run_eval_epoch(model, data, criterion, optimizer, batch_size, gradient_accumulation_steps, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_accuracy = 0
    batch_size = 1
    all_preds = []
    all_labels = []

    data_points = int(len(data["input_ids"]))
    step = 0

    with torch.no_grad():  # Disable gradient tracking
        for i in tqdm(range(0, data_points, batch_size)):
            step += 1

            input_ids = torch.stack(data["input_ids"][i:i + batch_size]).squeeze(1).to(device)
            attention_mask = torch.stack(data["attention_mask"][i:i + batch_size]).squeeze(1).to(device)
            labels = torch.Tensor(data["labels"][i:i + batch_size]).to(device)

            loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # Compute predictions
            preds = (torch.sigmoid(logits) > 0.5).long()
            accuracy = (preds == labels.long()).float().mean().item()

            total_loss += loss.item()
            total_accuracy += accuracy
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Averages
    avg_loss = total_loss / step
    avg_accuracy = total_accuracy / step

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)

    return avg_loss, avg_accuracy, precision, recall



def shuffle_data(data_dict):
    combined = list(zip(data_dict["input_ids"], data_dict["attention_mask"], data_dict["labels"]))
    random.shuffle(combined)
    input_ids, attention_mask, labels = zip(*combined)
    return {
        "input_ids": list(input_ids),
        "attention_mask": list(attention_mask),
        "labels": list(labels),
    }

def main(lr, wd):
    batch_size = 1
    device = torch.device("cuda:0")


    random.seed(12346474)
    np.random.seed(12346474)
    model_name = "google/long-t5-tglobal-base"
    print("Loading Data...")
    graphs, labels, _ = load_data()

    data = {
        "train" : graphs_to_t5_format(graphs["train"],  labels["train"]),
        "eval" : graphs_to_t5_format(graphs["eval"], labels["eval"]),
        "test" : graphs_to_t5_format(graphs["test"], labels["test"])
        }
    tokenizer = AutoTokenizer.from_pretrained(model_name)  

    print("Tokenizing Data...")

    train_inputs = tokenize_data(tokenizer, data['train'])
    eval_inputs = tokenize_data(tokenizer, data['eval'])
    test_inputs = tokenize_data(tokenizer, data['test'])

    train_inputs = shuffle_data(train_inputs)
    eval_inputs = shuffle_data(eval_inputs)
    test_inputs = shuffle_data(test_inputs)

    model = T5WithClassificationHead(model_name).to(device)    
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)  # Adjust weight_decay if needed

    best_epoch = 0
    best_eval_accuracy = 0
    best_eval_loss = float('inf')
    best_test_accuracy = 0
    best_test_loss = float('inf')
    stopped_early = False

    print("Training ")
    logging.info('train the model')
    print(len(train_inputs["input_ids"]))
    print(len(eval_inputs))
    train_start = datetime.now()
    for epoch in range(50):
        train_loss, train_accuracy = run_train_epoch(model=model, data=train_inputs, criterion=criterion, optimizer=optimizer, batch_size=batch_size, gradient_accumulation_steps=1, device=device)
        logging.info(f'train - {epoch = } # {train_loss = :.2f} # {train_accuracy = :.2f}')

        # get dev scores
        eval_loss, eval_accuracy, eval_precision, eval_recall = run_eval_epoch(model=model, data=eval_inputs, criterion=criterion, optimizer=optimizer, batch_size=batch_size, gradient_accumulation_steps=1, device=device)
        logging.info(f'dev   - {epoch = } # {eval_loss = :.2f} # {eval_accuracy = :.2f}')

        # get test scores
#        test_loss, test_accuracy, test_precision, test_recall = run_eval_epoch(model=model, data=test_inputs, criterion=criterion, optimizer=optimizer, batch_size=batch_size, gradient_accumulation_steps=1, device=device)
#        logging.info(f'test  - {epoch = } # {test_loss = :.2f} # {test_accuracy = :.2f}')

        if eval_loss < best_eval_loss:
            best_epoch = epoch
            best_eval_accuracy = eval_accuracy
            best_eval_loss = eval_loss
            best_eval_accuracy = eval_accuracy
            best_eval_loss = eval_loss

        wandb.log(
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "stopped_early": float(stopped_early),
                "train/accuracy": train_accuracy, "train/loss": train_loss, 
                "eval/accuracy": eval_accuracy, "eval/precision" : eval_precision, "eval/recall" : eval_recall, "eval/loss": eval_loss, 'eval/best_accuracy': best_eval_accuracy, 'dev/best_loss': best_eval_loss,
 #               "test/accuracy": test_accuracy, "test/precision" : test_precision, "test/recall" : test_recall, "test/loss": test_loss, 'test/best_accuracy': best_test_accuracy, 'test/best_loss': best_test_loss,
            }
        )

        last_epoch = epoch
        if epoch - best_epoch >= 5:
            logging.info(f'stopped early at epoch {epoch}')
            stopped_early = True
            break
    train_end = datetime.now()
    for epoch in range(last_epoch+1, 50):
        wandb.log(
            {
 		"epoch": epoch,
                "best_epoch": best_epoch,
                "stopped_early": float(stopped_early),
                "train/accuracy": train_accuracy, "train/loss": train_loss,
                "eval/accuracy": eval_accuracy, "eval/precision" : eval_precision, "eval/recall" : eval_recall, "eval/loss": eval_loss, 'eval/best_accuracy': best_eval_accuracy, 'dev/best_loss': best_eval_loss,
  #              "test/accuracy": test_accuracy, "test/precision" : test_precision, "test/recall" : test_recall, "test/loss": test_loss, 'test/best_accuracy': best_test_accuracy, 'test/best_loss': best_test_loss,
            }
        )
    inf_start = datetime.now()
    test_loss, test_accuracy, test_precision, test_recall = run_eval_epoch(model=model, data=test_inputs, criterion=criterion, optimizer=optimizer, batch_size=batch_size, gradient_accumulation_steps=1, device=device)
    inf_end = datetime.now()
    wandb.log(
        {
         "time/training": str(train_end-train_start), "time/inference": str(inf_end - inf_start),
         "test/accuracy": test_accuracy, "test/precision" : test_precision, "test/recall" : test_recall, 
        }
    )
if __name__ == "__main__":
    lr = 5e-5
    wd = 0
    run_id = 0

    for i in [1]:
        name = f"longT5-lr{lr}-wd{wd}-{i}"
        wandb_run = wandb.init(
            project="LongT5",
            name=name,
            config={
                "learning_rate": lr,
                "weight_decay": wd,
            }
        )
        logging.info(f"Running {name}")
        main(lr, wd)  # You can make batch_size and device configurable too
        logging.info("Done with main")

        wandb_run.finish()
        run_id += 1

