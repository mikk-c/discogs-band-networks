import numpy as np
import pandas as pd

def print_bloc(x):
   i = 0
   for index, row in x.iterrows():
      bb_parts = row['bb'].split('_')
      if i == 0:
         print(f"\\multirow{{6}}{{*}}{{{x.name.title()}}} & \\multirow{{2}}{{*}}{{{bb_parts[0]}}} & {bb_parts[1]} & ${row['mean']:.4f} \\pm {row['ci95']:.4f}$\\\\")
      elif i % 2 == 0:
         print(f"& \\multirow{{2}}{{*}}{{{bb_parts[0]}}} & {bb_parts[1]} & ${row['mean']:.4f} \\pm {row['ci95']:.4f}$\\\\")
      else:
         print(f"& & {bb_parts[1]} & ${row['mean']:.4f} \\pm {row['ci95']:.4f}$\\\\")
         if i != 5:
            print("\\noalign{\\vskip-4pt}\\cline{2-4}\\noalign{\\vskip-4pt}")
      i += 1
   print("\\hline")

df = pd.read_csv("auprcs.tsv", sep = "\t", header = None, names = ("task", "bb", "decorator", "quality")).fillna("")
df = df[df["bb"] != "mlp"]
df["bb"] = df.apply(lambda x: f"{x['bb']}_{x['decorator']}", axis = 1)

table = df.groupby(by = ["task", "bb"])["quality"].agg(["mean", "count", "std"]).reset_index()
table["ci95"] = 1.96 * table["std"] / np.sqrt(table["count"])

print(r"""
\begin{table}
\centering
\begin{tabular}{lll|r}
Task & Backbone & Strategy & AUPRC\\
\hline
""")

_ = table.groupby(by = "task").apply(print_bloc)

print(r"""
\end{tabular}
\caption{The area under the Precision-Recall curve (y axis) for different backbone strategy combinations. The table reports the average performance over 20 runs plus minus the 95\% confidence intervals. We highlight in bold the method performing the best per task. Multiple bold per task show top performances that cannot be statistically distinguished.}
\label{tab:val-pred}
\end{table}
""")
