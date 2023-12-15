import wandb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    api = wandb.Api()

    project = "gingerninja/DiversityGeneration"

    # Change according to env name used in tagging
    # env_prefix = "matrix-game-AHT-"
    # method_names = ["L-BRDiv", "BRDiv", "LIPO"]

    res_dict = {}
    final_data = []

    tag_name = "matrix-game-lbrdiv"
    runs = api.runs(project, filters={"tags": tag_name})
    m_name = "L-BRDiv"

    for run in runs:
        hist = run.scan_history()
        ret_list = [
            data["Train/sp/lagrange_mult_norm"] for data in hist if not data["Train/sp/lagrange_mult_norm"] is None
        ]
        print("Done done")
        if not tag_name in res_dict.keys(): 
            res_dict[tag_name] = []
        print(len(ret_list))
        concatenated_table = np.asarray([[m_name for _ in range(7500)], list(range(7500)), [(r - 0.2887) for r in ret_list]])
        res_dict[tag_name].append(concatenated_table)
        all_returns = np.concatenate(res_dict[tag_name], axis=-1)
        final_data.append(all_returns.T)

    all_returns_df = pd.DataFrame(np.concatenate(final_data, axis=0))
    all_returns_df.columns = ["Algorithm", "Total Timesteps (x960000)", "Returns Per Episode"]
    all_returns_df = all_returns_df.astype({"Total Timesteps (x960000)": np.int64, "Returns Per Episode": np.float64})

    g = sns.lineplot(data=all_returns_df, x="Total Timesteps (x960000)", y="Returns Per Episode", hue="Algorithm")
    g.get_legend().remove()
    sns.set_style('darkgrid')
    plt.title('Lagrange Multiplier Values in Repeated Matrix Game', fontdict={'fontsize': 16})
    plt.xlabel("Total Updates", fontsize=14)
    plt.ylabel("Lagrange Multiplier Mean Norm", fontsize=14)
    print("Preparing")
    plt.savefig("LagrangeMatrix.pdf")
