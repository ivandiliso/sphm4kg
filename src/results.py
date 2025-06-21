import pickle
import numpy as np
from tabulate import tabulate

file_name = "../out/KRKZEROONE.bin"

with open(file_name, "rb") as f:
    res_dict = pickle.load(f)



simple_prob, simple_rule = res_dict['simple']
hard_prob, hard_rule = res_dict['hard']

models = {
    'LogTwo' : None,
    'LogOne': None
}



table = []
headers = ["Dataset", "Model", "Type", "Precision", "Recall", "F1", "AUC"]

simple_averages, simple_rule_averages = res_dict["simple"]
hard_averages, hard_rule_averages = res_dict["hard"]


averages = np.concatenate([simple_averages, hard_averages], axis=2)
rule_averages = np.concatenate([simple_rule_averages, hard_rule_averages], axis=2)

for m, m_name in enumerate(models.keys()):
    for sel_score, sel_name in zip([averages, rule_averages], ["Prob", "Rule"]):

        if (m_name in ["LogOne", "LogTwo", "RIPPER"]) and (sel_name == "Rule"):
            continue

        row = [
            file_name,
            m_name,
            sel_name,
            f"{sel_score[m][0].mean():1.3f} ± {sel_score[m][0].std():1.3f}",
            f"{sel_score[m][1].mean():1.3f} ± {sel_score[m][1].std():1.3f}",
            f"{sel_score[m][2].mean():1.3f} ± {sel_score[m][2].std():1.3f}",
            f"{sel_score[m][3].mean():1.3f} ± {sel_score[m][3].std():1.3f}",
        ]

        table.append(row)

print(tabulate(table, headers=headers, tablefmt="github"))         