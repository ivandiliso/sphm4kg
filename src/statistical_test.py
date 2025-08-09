import numpy as np
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from pathlib import Path
import pickle


res_path = Path().cwd().absolute() / "out" / "dbpedia_parsed.bin"

with open(res_path, "rb") as f:
    raw_data = pickle.load(f)


method_names = ["MB", "H_VB", "H_EM", "H_GD", "T", "L", "HL"]
metric_names = ["Precision", "Recall", "F1", "AUC"]

for i in [0,1,2,3]:
    print(f"{80*'#'}")
    print(f"# Statistical Test on {metric_names[i].upper()}")
    s_avg, _ = raw_data["simple"]
    h_avg, _ = raw_data["hard"]

    s_avg = s_avg[:,i,:]
    h_avg = h_avg[:,i,:]

    data = np.hstack([s_avg, h_avg]).T

    stat, p = friedmanchisquare(*[data[:,i] for i in range(data.shape[1])])
    print(f"Friedman test statistic = {stat:.4f}, p-value = {p:.4f}")

    if p < 0.05:
        print("Significant differences detected by Friedman test.")
        # Nemenyi post-hoc test
        
        nemenyi_results = sp.posthoc_nemenyi_friedman(data)
        nemenyi_results.index = method_names
        nemenyi_results.columns = method_names
        print("\nNemenyi post-hoc test p-values:")
        print(nemenyi_results)
    else:
        print("No significant differences detected by Friedman test.")
    print("")
 










