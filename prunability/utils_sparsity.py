from typing import Dict
import json

from fflib.enums import SparsityType

def get_sparsity(network_type: str, report: Dict):
    result = []
    for key in report.keys():
        hoyer = report[key]['HOYER']
        if network_type == 'ff':
            layer1 = hoyer['layer_0']
            layer2 = hoyer['layer_1']
        if network_type == 'ffc':
            layer1 = hoyer['layer_0']
            layer2 = hoyer['layer_1']
        if network_type == 'ffrnn':
            layer1 = hoyer['layer_1']['fw+bw']
            layer2 = hoyer['layer_2']['fw+bw']
        if network_type == 'bp':
            layer1 = hoyer['layer0']['weight']
            layer2 = hoyer['layer1']['weight']
        result.append((layer1, layer2))
    return result

def read_sparsity_report(file: str, sparsity_type: SparsityType):
    with open(file, "r") as f:
        data = f.read()
        dict = json.loads(data)

        result = { }
        for type in SparsityType:
            result[str(type).split('.')[1]] = []
        
        for epoch in dict.keys():
            for batch in dict[epoch]:
                for type in SparsityType:
                    t = str(type).split('.')[1]
                    result[t].append(batch[t])
        
        t = str(sparsity_type).split('.')[1]
        return result[t]

