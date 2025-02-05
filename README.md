# A Novel Robust Integrating Method by High-order Proximity for Self-supervised Attribute Network Embedding

## Abstract
![framework](https://github.com/user-attachments/assets/50c20bd0-14ed-4b94-8cd4-348dc6829c7a)
**Fig. 1**: The general framework of RSANE consists of three key components: heterogeneous information integration, joint embedding of structures and attributes, and adaptive outlier resistance. Given an attribute network $\mathcal{G}=(\mathbf{A},\mathbf{X})$ where $\mathbf{A}$ is the adjacency matrix and $\mathbf{X}$ is the attribute matrix, topological and semantic information are extracted by first-order proximity and cosine similarity respectively. The integrated weights $\mathbf{Q}$ sum up the topological and semantic information and multiply it by the high-order proximity. It derives three weights to constrain $\mathbf{H}$ in the embedding space, and $\hat{\mathbf{A}}$ and $\hat{\mathbf{X}}$ in the reconstruction space. Loss $\mathcal{L}\_{RSANE\}$ is calculated based on the outlier weights $\varphi_i$, and the outlier scores $\phi_i$ will be updated based on $\mathcal{L}_\{RSANE\}$.

## Dependencies

> matplotlib==3.8.0<br>
numpy==2.1.3<br>
scikit_learn==1.5.2<br>
torch==2.5.1+cu121<br>
torch_geometric==2.6.1


## Dataset
In `./data/`, the `WebKB`,`Cora`, `CiteSeer`, `Amazon` and `Twitch` datasets are provided, along with the corresponding processed versions for link prediction and outlier detection.


## Training
Detailed training results and configs for node classification, link prediction, attribute prediction, outlier detection and network visualization are provided in `./Results.ipynb`.

Besides, it is also easy to run `./classification.py` directly to perform node classification, as is the case for the other graph learning tasks.


## Citation
If you find the code useful for your research, we kindly request to consider citing our work:
>@article{wu2025novel,
  title={A novel robust integrating method by high-order proximity for self-supervised attribute network embedding},
  author={Wu, Zelong and Wang, Yidan and Hu, Kaixia and Lin, Guoliang and Xu, Xinwei},
  journal={Expert Systems with Applications},
  volume={266},
  pages={125911},
  year={2025},
  publisher={Elsevier}
}
