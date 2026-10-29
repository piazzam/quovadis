# QUO VADIS - PhD Thesis Experiments

Source code for the image-classification experiments of the PhD thesis chapter **“QUO VADIS: Distributed Attribution for Image Classifiers.”**

The repository contains the QUO VADIS implementation, the baseline explainers used in the chapter, evaluation metrics, the trained low-resolution checkpoints, the ImageNet 10K sampling indices, and experiment runners for both the thesis experiments and additional analyses of the ensemble construction.

## Experiments covered

### Datasets

- MNIST
- Fashion-MNIST
- CIFAR-10
- CIFAR-100
- ImageNet (10 images per class, 10,000 images total)

### Attribution methods

- QUO VADIS
- PartitionSHAP
- SHAP-BPT
- LIME
- Occlusion
- Integrated Gradients
- Grad-CAM
- Grad-CAM++
- Grad-CAM-ViT for ImageNet

### Metrics

- ROAD-MoRF
- ROAD-LeRF
- SaCo
- Infidelity
- Max-Sensitivity
- Complexity

### QUO VADIS ablations

- feature removal: average blur, Gaussian blur, black replacement;
- solution concept: normalized Banzhaf, Shapley;
- aggregation: median, 25th quantile, contamination-aware;
- template configurations: A, B, C.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements-gpu.txt
├── requirements-cifar100.txt
├── configs/
│   ├── mnist.json
│   ├── fashion_mnist.json
│   ├── cifar10.json
│   ├── cifar100.json
│   └── imagenet.json
├── checkpoints/
│   ├── mnist_model.h5
│   ├── fashion_mnist_model.h5
│   └── cifar10_model.h5
├── data/
│   └── imagenet_sample_indices.csv
├── scripts/
│   └── create_imagenet_sample.py
├── experiments/
│   ├── run_suite.py
│   └── synthetic_leakage.py
├── src/quovadis_repro/
├── tests/
├── generate_explanations.py
├── evaluate_explanations.py
└── aggregate_quovadis.py
```

## Installation

Run all commands from the repository root.

### Main environment

This environment is intended for MNIST, Fashion-MNIST, CIFAR-10, and ImageNet.

```bash
conda create -n quovadis-thesis python=3.10 pip -y
conda activate quovadis-thesis
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
python -m pip install -e . --no-deps
```

### CIFAR-100 environment

CIFAR-100 used a separate PyTorch software stack in the original experiments.

```bash
conda create -n quovadis-cifar100 python=3.9 pip -y
conda activate quovadis-cifar100
python -m pip install --upgrade pip
python -m pip install -r requirements-cifar100.txt
python -m pip install -e . --no-deps
```

### Lightweight package installation

The package also exposes optional dependency groups:

```bash
pip install -e '.[tensorflow]'
pip install -e '.[torch]'
pip install -e '.[shap-bpt]'
```

or all optional dependencies:

```bash
pip install -e '.[all]'
```

SHAP-BPT is pinned to `shap-bpt==1.0.1`, matching the implementation used for the thesis experiments. The local wrapper includes the NumPy compatibility workaround required by that release.

## Verify the installation

```bash
python -c "import quovadis_repro; print('QUO VADIS import: OK')"
python -m pytest -q
```

The unit tests cover the quotient-game solution concepts, template generation, feature-removal operators, and SHAP-BPT conversion helpers without requiring a full experiment run.

## Dataset and model setup

### MNIST, Fashion-MNIST, CIFAR-10

The trained Keras checkpoints are included under `checkpoints/`. Dataset loading is handled automatically by the configuration files.

### CIFAR-100

The configuration uses the pre-trained ResNet-50 setup used in the experiments. The required packages are listed in `requirements-cifar100.txt`.

### ImageNet

The repository contains the exact source-row indices for the 10,000-image validation subset:

```text
data/imagenet_sample_indices.csv
```

The serialized image subset is not included. Reconstruct it from the gated `ILSVRC/imagenet-1k` validation split after obtaining access on Hugging Face:

```bash
hf auth login
python scripts/create_imagenet_sample.py
```

The resulting file is:

```text
data/imagenet1k_val_sampled10K.parquet
```

`configs/imagenet.json` already points to this path.

## Core workflow

Explanation generation and evaluation are separated. This makes it possible to generate expensive attribution maps once and evaluate them multiple times.

### Generate QUO VADIS explanations

```bash
python generate_explanations.py \
  --config configs/mnist.json \
  --method quovadis \
  --output outputs/mnist/quovadis
```

The default configuration is:

- Configuration A;
- normalized Banzhaf index;
- median aggregation;
- average blur;
- `11 x 11` blur kernel.

For a quick smoke test:

```bash
python generate_explanations.py \
  --config configs/mnist.json \
  --method quovadis \
  --limit 10 \
  --output outputs/smoke/mnist_quovadis
```

### Evaluate saved explanations

```bash
python evaluate_explanations.py \
  --config configs/mnist.json \
  --method quovadis \
  --explanations outputs/mnist/quovadis \
  --metrics road_morf road_lerf saco infidelity sensitivity complexity \
  --output results/mnist/quovadis
```

`road_morf` and `road_lerf` use the same ROAD perturbation/imputation pipeline and differ only in the attribution ordering used to select removed pixels. The legacy metric name `road` is still accepted and follows `metrics.road.direction` from the JSON configuration.

### ROAD-MoRF and ROAD-LeRF

Both ROAD directions can be evaluated from the same saved attribution maps:

```bash
python evaluate_explanations.py \
  --config configs/cifar10.json \
  --method quovadis \
  --explanations outputs/cifar10/quovadis \
  --metrics road_morf road_lerf \
  --output results/cifar10/quovadis_road
```

The corresponding outputs are `road_morf_results.npz` and `road_lerf_results.npz`. Each file stores the removal percentages, accuracy curve, per-image correctness matrix, and AUC.

Sensitivity is considerably more expensive than the other metrics because explanations are regenerated for perturbed inputs. It can be run separately:

```bash
python evaluate_explanations.py \
  --config configs/mnist.json \
  --method quovadis \
  --explanations outputs/mnist/quovadis \
  --metrics sensitivity \
  --output results/mnist/quovadis_sensitivity
```

## Baseline methods

The same generation interface is used for every baseline.

Examples:

```bash
python generate_explanations.py --config configs/mnist.json --method partition_shap --output outputs/mnist/partition_shap
python generate_explanations.py --config configs/mnist.json --method shap_bpt --output outputs/mnist/shap_bpt
python generate_explanations.py --config configs/mnist.json --method integrated_gradients --output outputs/mnist/integrated_gradients
python generate_explanations.py --config configs/mnist.json --method lime --output outputs/mnist/lime
python generate_explanations.py --config configs/mnist.json --method occlusion --output outputs/mnist/occlusion
python generate_explanations.py --config configs/mnist.json --method gradcam --output outputs/mnist/gradcam
python generate_explanations.py --config configs/mnist.json --method gradcampp --output outputs/mnist/gradcampp
```

For ImageNet, use `gradcam_vit` instead of the CNN Grad-CAM variants:

```bash
python generate_explanations.py \
  --config configs/imagenet.json \
  --method gradcam_vit \
  --output outputs/imagenet/gradcam_vit
```

### SHAP-BPT

SHAP-BPT is integrated directly into the common explainer interface for TensorFlow and PyTorch. The default settings in all dataset configurations are:

```json
{
  "max_evals": 254,
  "baseline": 0.0,
  "method": "BPT",
  "balance_area": false,
  "min_area": 1
}
```

The wrapper fixes the explained output to the requested target class, which is the model top-1 prediction in the standard generation pipeline.

## QUO VADIS feature-removal strategies

### Average blur

```bash
python generate_explanations.py \
  --config configs/cifar10.json \
  --method quovadis \
  --removal-strategy average_blur \
  --output outputs/cifar10/qv_average_blur
```

### Gaussian blur

The implementation matches the experimental operator based on:

```python
cv2.GaussianBlur(image, (11, 11), 0)
```

Run it with:

```bash
python generate_explanations.py \
  --config configs/cifar10.json \
  --method quovadis \
  --removal-strategy gaussian_blur \
  --output outputs/cifar10/qv_gaussian_blur
```

### Black replacement

```bash
python generate_explanations.py \
  --config configs/cifar10.json \
  --method quovadis \
  --removal-strategy black \
  --black-value 0 \
  --output outputs/cifar10/qv_black
```

## QUO VADIS solution concept

Normalized Banzhaf:

```bash
python generate_explanations.py \
  --config configs/cifar10.json \
  --method quovadis \
  --solution banzhaf \
  --output outputs/cifar10/qv_banzhaf
```

Shapley:

```bash
python generate_explanations.py \
  --config configs/cifar10.json \
  --method quovadis \
  --solution shapley \
  --output outputs/cifar10/qv_shapley
```

## Aggregation strategies

Median:

```bash
python generate_explanations.py --config configs/cifar10.json --method quovadis --aggregation median --output outputs/cifar10/qv_median
```

25th quantile:

```bash
python generate_explanations.py --config configs/cifar10.json --method quovadis --aggregation q25 --output outputs/cifar10/qv_q25
```

Contamination-aware:

```bash
python generate_explanations.py --config configs/cifar10.json --method quovadis --aggregation contamination_aware --output outputs/cifar10/qv_contamination_aware
```

## Template configurations A, B, C

```bash
python generate_explanations.py --config configs/cifar10.json --method quovadis --template-configuration A --output outputs/cifar10/qv_A
python generate_explanations.py --config configs/cifar10.json --method quovadis --template-configuration B --output outputs/cifar10/qv_B
python generate_explanations.py --config configs/cifar10.json --method quovadis --template-configuration C --output outputs/cifar10/qv_C
```

Configuration A contains 17 templates. B and C are the subsets used in the thesis ablation.

## Running complete experiment suites

`experiments/run_suite.py` generates explanations and evaluates them in one command. It writes each variant to its own directory and records a machine-readable `manifest.json`.

All examples below can be restricted during development with `--limit N`. You can also use `--skip-existing` to resume a partially completed suite.

### Main baseline comparison

MNIST example:

```bash
python experiments/run_suite.py baselines \
  --config configs/mnist.json \
  --output results/mnist
```

For ImageNet the runner automatically uses Grad-CAM-ViT instead of Grad-CAM/Grad-CAM++.

To run only selected methods:

```bash
python experiments/run_suite.py baselines \
  --config configs/imagenet.json \
  --methods quovadis partition_shap shap_bpt gradcam_vit \
  --output results/imagenet
```

### Feature-removal ablation

Runs average blur, Gaussian blur, and black replacement:

```bash
python experiments/run_suite.py feature-removal \
  --config configs/cifar10.json \
  --output results/cifar10
```

### Banzhaf versus Shapley

```bash
python experiments/run_suite.py solution \
  --config configs/cifar10.json \
  --output results/cifar10
```

### Median, 25th-quantile, and contamination-aware aggregation

```bash
python experiments/run_suite.py aggregation \
  --config configs/cifar10.json \
  --output results/cifar10
```

### Template configurations A, B, C

```bash
python experiments/run_suite.py configurations \
  --config configs/cifar10.json \
  --output results/cifar10
```

## Additional ensemble experiments

These experiments are intended to isolate properties of the ensemble construction rather than reproduce a publication table.

### Single template versus the complete ensemble

Runs each of the 17 Configuration-A templates independently and then the full Configuration-A ensemble:

```bash
python experiments/run_suite.py single-template \
  --config configs/cifar10.json \
  --output results/cifar10_extensions
```

### Number of templates

By default the runner evaluates `R = 1, 2, 4, 8, 12, 17`. For every size below 17, five random template subsets are sampled using seed 42.

```bash
python experiments/run_suite.py template-count \
  --config configs/cifar10.json \
  --repetitions 5 \
  --subset-seed 42 \
  --output results/cifar10_extensions
```

Custom values:

```bash
python experiments/run_suite.py template-count \
  --config configs/cifar10.json \
  --template-counts 1 3 5 10 17 \
  --repetitions 10 \
  --output results/cifar10_extensions
```

### Granularity / number of quotient-game players

Templates are grouped by their number of players, and each group is aggregated independently:

```bash
python experiments/run_suite.py granularity \
  --config configs/cifar10.json \
  --output results/cifar10_extensions
```

The available groups in Configuration A contain 2, 3, 5, 6, 7, or 8 players.

### Robustness to template selection

Samples multiple random subsets with the same `R`, making it possible to measure how much the final performance depends on which templates are chosen:

```bash
python experiments/run_suite.py template-robustness \
  --config configs/cifar10.json \
  --subset-size 8 \
  --repetitions 20 \
  --subset-seed 42 \
  --output results/cifar10_extensions
```

### Faster exploratory runs

Any suite can evaluate a subset of metrics:

```bash
python experiments/run_suite.py single-template \
  --config configs/cifar10.json \
  --metrics road_morf road_lerf saco complexity \
  --limit 100 \
  --output results/debug
```

## Synthetic attribution-leakage experiment

This experiment does not require a classifier. It generates additive image games with known pixel contributions `w_p`, applies all 17 Configuration-A partitions, uniformly assigns each group contribution to its pixels, and measures the resulting pixel-level error. It also evaluates median and 25th-quantile ensemble aggregation.

```bash
python experiments/synthetic_leakage.py \
  --output results/synthetic_leakage \
  --trials 100 \
  --height 32 \
  --width 32 \
  --distribution sparse_signed \
  --sparsity 0.1 \
  --seed 42
```

Available synthetic distributions are:

- `normal`;
- `sparse_positive`;
- `sparse_signed`.

The script stores:

```text
synthetic_leakage.npz
summary.json
```

The NPZ contains the error for every template and trial, together with the ensemble errors.

## Explicit template subsets

The standard generator can also be used with an arbitrary subset of Configuration-A templates:

```bash
python generate_explanations.py \
  --config configs/cifar10.json \
  --method quovadis \
  --templates 5_ellipse 5_row 5_column 6_block \
  --output outputs/cifar10/custom_subset
```

This is the mechanism used by the additional ensemble experiments.

## Output format

Each generated explanation directory contains at least:

```text
explanations.npy
metadata.json
generation_summary.json
```

Attribution arrays use shape:

```text
(N, H, W)
```

and always follow dataset order:

```text
explanations[i] <-> dataset[i]
```

Evaluation directories contain raw metric values such as:

```text
road_morf_results.npz
road_lerf_results.npz
saco_scores.npy
infidelity_scores.npy
sensitivity_scores.npy
complexity_scores.npy
summary.json
```

Experiment suites additionally write `run.json` for each variant and a suite-level `manifest.json`.

## Reproducibility conventions

- Default random seed: `42`.
- Explanations target the model-predicted class.
- TensorFlow images use channel-last layout.
- PyTorch CIFAR-100 and ImageNet images use channel-first layout after preprocessing.
- QUO VADIS average and Gaussian blur use an `11 x 11` kernel.
- Black replacement uses value `0.0` by default.
- PartitionSHAP uses its configured `16 x 16` blur masker.
- Occlusion uses the dataset configuration, normally a `16 x 16` window and stride `4`.
- SHAP-BPT uses `max_evals=254` by default.

## Notes on computational cost

Several complete suites are expensive, particularly on ImageNet. `Sensitivity` is especially costly because it recomputes explanations for perturbed inputs. For development, first use `--limit` and a reduced metric list; remove the limit only for the final run.

The experiment runner is resumable with `--skip-existing`:

```bash
python experiments/run_suite.py baselines \
  --config configs/imagenet.json \
  --output results/imagenet \
  --skip-existing
```

## Research context and supervision

This work was developed as part of a PhD thesis at the **MIND Laboratory**, Department of Informatics, Systems and Communication (DISCo), **University of Milano-Bicocca**, under the supervision of **Prof. Enza Messina** and **Prof. Mauro Passacantando**.

## Acknowledgements

Special thanks to **Antonio Consolo** for his valuable contribution to this work.

## License

This repository is released under the **MIT License**. See [`LICENSE`](LICENSE) for the full licence text.
