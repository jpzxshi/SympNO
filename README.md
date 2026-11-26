# SympNO

**Title:** SympNO: Intrinsic structure-preserving symplectic neural operator for Hamiltonian PDEs

**Authors:** Weijun Zhang, Chenyu Zeng, Jitong Zhu, Daniel Zhengyu Huang, Pengzhan Jin

**Preprint:** [https://doi.org/10.5281/zenodo.24560672](https://doi.org/10.5281/zenodo.24560672)

---

```text
SympNO_code/
├── sympno.py
├── training.py
├── wave1d/
│   ├── main.py
│   └── data.py
├── wave2d/
│   ├── main.py
│   └── data.py
├── schrodinger/
│   ├── main.py
│   └── data.py
├── kleingordon/
│   ├── main.py
│   └── data.py
├── requirements.txt
└── README.md
```

`sympno.py` defines the network shared by all four experiments. `training.py`
handles training, validation, and testing. Each experiment has a `main.py`
with its parameters and a `data.py` for initial conditions and reference solutions.

## Run

Requires Python 3.10+, NumPy, and PyTorch. Install
[PyTorch](https://pytorch.org/get-started/locally/) for your hardware, then install
the dependencies:

```bash
pip install -r requirements.txt
```

Run an experiment:

```bash
python wave1d/main.py
python wave2d/main.py
python schrodinger/main.py
python kleingordon/main.py
```

Set `data_config`, `model_config`, and `train_config` in each `main.py`.
Data generation uses NumPy on the CPU. Training uses CUDA when available.

## Data

Download the datasets from PKU cloud storage. All links are permanent.

| Experiment | Download | Extract into |
|---|---|---|
| Wave1D | [wave1d_data.zip](https://disk.pku.edu.cn/link/AAFB99FD86E6CB4E9B953AB172EA520993) | `wave1d/` |
| Wave2D | [wave2d_data.zip](https://disk.pku.edu.cn/link/AAC8716A0AEAC34FA39F0A94783BB6D3CB) | `wave2d/` |
| Schrodinger | [schrodinger_data.zip](https://disk.pku.edu.cn/link/AA841BCF7D4C1B4870A1C9D759518C0FB5) | `schrodinger/` |
| Kleingordon | [kleingordon_data.zip](https://disk.pku.edu.cn/link/AA69823433CB584E78BB2E48B4FB86E930) | `kleingordon/` |

Each archive contains a `dataset/` folder with `inputs.npy` and `targets.npy`.
Extract it into the corresponding experiment folder. For example, extracting
`wave1d_data.zip` into `wave1d/` gives `wave1d/dataset/inputs.npy` and
`wave1d/dataset/targets.npy`.

No metadata file is needed. Existing arrays are loaded directly. 

**We strongly recommend downloading the datasets using the links above directly.** 
If the datasets cannot be downloaded successfully, data generation scripts are also available for all four experiments.
For example, generate the Wave1D dataset with:

```bash
python wave1d/data.py
```

Existing arrays are not checked against the parameters or validated for completeness.
After changing data parameters, regenerate the data or change `data_path`.
If generation is interrupted, rerun `data.py` and let it finish before training.

| Experiment | Grid | State pairs | dt | Data seed | Reference solver |
|---|---|---:|---:|---:|---|
| Wave1D | 100 | 100000 | 0.001 | 42 | Exact Fourier flow |
| Wave2D | 128 × 128 | 10000 | 0.01 | 42 | Eighth-order Yoshida splitting, 4 substeps |
| Schrodinger | 100 | 100000 | 0.001 | 42 | Exact Fourier flow |
| Kleingordon | 100 | 100000 | 0.001 | 42 | Eighth-order Yoshida splitting, 4 substeps |

All spatial directions use the periodic interval `[0, 1)`. States are stored
in float64 and flattened as `[q, p]`. For Schrodinger, q and p are the real
and imaginary parts of the complex field.

Each 1D sample contains an independent random input field and its state after
one time step. Wave2D uses 200 trajectories with 50 pairs per trajectory:
input steps are `0, 20, ..., 980`, and target steps are `1, 21, ..., 981`.
Each 1D dataset takes about 0.32 GB; Wave2D takes about 5.24 GB.
Wave2D writes one pair at a time to limit memory use. Full CPU generation
is slow, so downloading the data is an alternative.

## Training

All four experiments use 4 SympNO layers, width 256, GELU activation, and
float64 parameters. The 1D models use 32 Fourier modes and spatial MLPs
of size `4 → 16 → 256`. The 2D model uses `(16, 16)` Fourier modes and
spatial MLPs of size `8 → 32 → 256`. FFTs use complex128.

Training runs for 500 epochs with batch size 8 and state MSE over one time step.
Adam uses `lr=1e-3`, `betas=(0.9, 0.999)`, and no weight decay.
Wave1D uses `eps=1e-14`; the other experiments use `eps=1e-12`.
All four experiments use `fused=True` on CUDA.
The learning rate follows a cosine schedule down to approximately `1e-5`.
Wave2D starts with 5 warmup epochs.

All random seeds are 42. Samples are split into 80% training, 10% validation,
and 10% test data. The full validation set is evaluated at epoch 0, every
5 epochs, and at the final epoch. Evaluation batch sizes are 10000 for the
three 1D experiments and 128 for Wave2D. The checkpoint with the lowest
validation MSE is saved and then evaluated on the test set.

Results are saved in each experiment's `outputs/` directory:

- `best.pt`: the state dictionary of the model with the lowest validation MSE.
- `loss.txt`: epoch, learning rate, training MSE, and validation MSE; `nan` marks epochs without evaluation.
- `test.txt`: the selected epoch, validation MSE, and test MSE.

Each run trains from scratch and overwrites the results in its output directory.
Change `output_path` in `main.py` to keep results from multiple runs.
