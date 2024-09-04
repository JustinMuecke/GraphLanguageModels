import yaml
import json
from typing import List
triples : List[List[str]] = []

with open("test.yaml", "r") as file: 

    lines = file.readlines()

    remove = []
    for i in range(0,len(lines)):
        lines[i] = lines[i].strip().replace("\n", "")
        if "and " in lines[i]:
            lines[i-1] = lines[i-1] + " " + lines[i]
            remove.append(i)

    for index in remove: 
        lines.pop(index)

    line = lines[0]
    for i in range(1, len(lines)):


        line2 = lines[i]
        

        if(line == "" and line2 == ""):
            line = lines[i]
            continue

        
        if(line == ""):
            line = lines[i]
            continue

        tuple = line.split(":")
        if(":" in line):
            if(tuple[1] != ""):
                obj = tuple[0]
                sub = tuple[1]
                relation = "is"
            else: 
                relation = tuple[0].strip()
                line = lines[i]
                continue
        else:
            obj = line.strip()        

        line = lines[i].replace("\n", "").strip()
        triples.append([sub, relation, obj])



for i in range(len(triples)):
    [sub, relation, obj] = triples[i]
    if(relation == "Facts"):
        tuple = obj.split("  ")
        relation = tuple[0]
        obj = tuple[1]
    triples[i] = [sub, relation, obj]
print(triples)

with open("data.jsonl", "w") as file: 
    json.dump(triples, file)