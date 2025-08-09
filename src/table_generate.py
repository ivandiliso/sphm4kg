
import pickle
from pathlib import Path
import numpy as np
from tabulate import tabulate


models = [
    'MBNB',
    'HB_VB',
    'HB_EM',
    'HB_GD',
    'Tree',
    "HLogReg",
    "LogREg"
]

res_path = Path().cwd().absolute() / "out" / "dbpedia_parsed.bin"

with open(res_path, "rb") as f:
    raw_data = pickle.load(f)


simple_avg, simple_rule_avg = raw_data["simple"]
hard_avg, hard_rule_avg = raw_data["hard"]



averages = np.concatenate([simple_avg, hard_avg], axis=2)
rule_averages = np.concatenate([simple_rule_avg, hard_rule_avg], axis=2)



table = []
headers = ["Dataset", "Model", "Type", "Precision", "Recall", "F1", "AUC"]



for m, m_name in enumerate(models):
    for sel_score, sel_name in zip([averages, rule_averages], ["Prob", "Rule"]):

        row = [
            "dbpedia",
            m_name,
            sel_name,
            f"{sel_score[m][0].mean():1.3f} ± {sel_score[m][0].std():1.3f}",
            f"{sel_score[m][1].mean():1.3f} ± {sel_score[m][1].std():1.3f}",
            f"{sel_score[m][2].mean():1.3f} ± {sel_score[m][2].std():1.3f}",
            f"{sel_score[m][3].mean():1.3f} ± {sel_score[m][3].std():1.3f}",
        ]

        table.append(row)

print(tabulate(table, headers=headers, tablefmt="github"))