import torch
import matplotlib.pyplot as plt
import numpy as np
import os
# d = ["p0.1", "10farest","mean","offdropout"]
# obss = [torch.load(f"/raid1/p3/jiahao/view_sampling/desimcse/result/simcse-bert-base-uncased-{dd}-new_var/obs.pt") for dd in d]
# output_path = "fig/agg/"


# d = [0.01, 0.1, 0.2, 0.3]
# obss = [torch.load(f"/raid1/p3/jiahao/view_sampling/desimcse/result/simcse-bert-base-uncased-p{dd}-new_var/obs.pt") for dd in d]
# output_path ="fig/dropout"


# d = [0.2,0.3,0.4]
# obss = [torch.load(f"/raid1/p3/jiahao/view_sampling/desimcse/result/simcse-bert-base-uncased-offdropout-pos-1-drop_p{dd}-new_var/obs.pt") for dd in d]
# output_path = "fig/pos_1/"
# d.append(0)
# obss.append(torch.load("/raid1/p3/jiahao/view_sampling/desimcse/result/simcse-bert-base-uncased-offdropout/obs.pt"))

d = [0.8,0.9,1,1.1]
obss = [torch.load(f"/raid1/p3/jiahao/view_sampling/desimcse/result/simcse-bert-base-uncased-offdropout-sw0.1-sw_offdropout{dd}-new_var/obs.pt") for dd in d]
output_path = "fig/offdropout_sw_offdropout_pos_ratio/"
# d.append(0)
# obss.append(torch.load("/raid1/p3/jiahao/view_sampling/desimcse/result/simcse-bert-base-uncased-offdropout-sw0.1-sw_None-new_var/obs.pt"))


# obss.append(torch.load("/raid1/p3/jiahao/view_sampling/desimcse/result/my-unsup-simcse-bert-base-uncased-0.1-mean-z1-z2/obs.pt"))
# obss.append(torch.load("/raid1/p3/jiahao/view_sampling/desimcse/result/re-unsup-simcse-bert-base-uncased-pos_ratio_0.9-fp16/obs.pt"))
# d.append("mean")
# d.append("de")
step=3000
intervals=10

if not os.path.exists(output_path):
    os.makedirs(output_path)


with plt.style.context('science'):
    plt.figure(figsize=(6,5))
    for i,obs in enumerate(obss):
        plt.plot(list(range(0, step, intervals)), np.log(np.array(obs["grad_diff"][:step:intervals])), label=f"{d[i]}")
    plt.legend()
    plt.xlabel("steps")
    plt.ylabel("submodel gradient distance (log10 L2)")
    plt.savefig(output_path+"/train_gradient.pdf",dpi=1200)
    plt.show()

with plt.style.context('science'):
    plt.figure(figsize=(6,5))
    for i,obs in enumerate(obss):
        plt.plot(list(range(0, step, intervals)), np.log(np.array(obs["align"][:step:intervals])+1e-3), label=f"{d[i]}")
    plt.legend()
    plt.xlabel("steps")
    plt.ylabel("submodel alignment loss (log10)")
    plt.savefig(output_path+"/train_align.pdf", dpi=1200)
    plt.show()

with plt.style.context('science'):
    plt.figure(figsize=(6,5))
    for i,obs in enumerate(obss):
        plt.plot(list(range(0, step, intervals)), np.array(obs["uni"][:step:intervals])+1e-3, label=f"{d[i]}")
    plt.legend()
    plt.xlabel("steps")
    plt.ylabel("submodel uniformity loss ")
    plt.savefig(output_path+"/train_uni.pdf", dpi=1200)
    plt.show()

with plt.style.context('science'):
    plt.figure(figsize=(6,5))
    for i,obs in enumerate(obss):
        plt.plot(list(range(0, step, intervals)), np.log2(np.array(obs["pos_sim_var"][:step:intervals])), label=f"{d[i]}")
    plt.legend()
    plt.xlabel("steps")
    plt.ylabel("pos sim var ")
    plt.savefig(output_path+"/pos_sim_var.pdf",dpi=1200)
    plt.show()
with plt.style.context('science'):
    plt.figure(figsize=(6,5))
    for i,obs in enumerate(obss):
        plt.plot(list(range(0, step, intervals)), np.log2(np.array(obs["neg_sim_var"][:step:intervals])), label=f"{d[i]}")
    plt.legend()
    plt.xlabel("steps")
    plt.ylabel("neg sim var ")
    plt.savefig(output_path+"/neg_sim_var.pdf",dpi=1200)
    plt.show()