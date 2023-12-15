import wandb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    api = wandb.Api()

    project = "gingerninja/DiversityGeneration"

    # Change according to env name used in tagging
    env_prefix = "matrix-game-AHT-"
    method_names = ["L-BRDiv", "BRDiv", "LIPO"]

    res_dict = {}
    final_data = []
    for m_name in method_names:
        tag_name = env_prefix + m_name
        #tag_name = "small-lbf-blinded-lbrdiv"
        runs = api.runs(project, filters={"tags": tag_name})

        for run in runs:
            #print("RUNNING!")
            hist = run.scan_history()
            ret_list = [
                data["Returns/generalise/nondiscounted"] for data in hist if not data["Returns/generalise/nondiscounted"] is None
            ]
            if not tag_name in res_dict.keys(): 
                res_dict[tag_name] = []
            concatenated_table = np.asarray([[m_name for _ in range(51)], list(range(51)), ret_list])
            res_dict[tag_name].append(concatenated_table)
        all_returns = np.concatenate(res_dict[tag_name], axis=-1)
        final_data.append(all_returns.T)

    all_returns_df = pd.DataFrame(np.concatenate(final_data, axis=0))
    all_returns_df.columns = ["Algorithm", "Total Timesteps (x960000)", "Returns Per Episode"]
    all_returns_df = all_returns_df.astype({"Total Timesteps (x960000)": np.int64, "Returns Per Episode": np.float64})

    sns.lineplot(data=all_returns_df, x="Total Timesteps (x960000)", y="Returns Per Episode", hue="Algorithm")
    sns.set_style('darkgrid')
    plt.title('Generalization Performance in Repeated Matrix Game', fontdict={'fontsize': 16})
    plt.xlabel("Total Timesteps (x20000)", fontsize=14)
    plt.ylabel("Returns Per Episode", fontsize=14)
    plt.savefig("MatrixGame.pdf")
