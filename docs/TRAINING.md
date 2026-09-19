# Historical training / 历史训练说明

## Current deployment is not a trained model

当前v2 MCP不加载历史权重，也不运行训练。Docker只安装CPU版数值/文档工具依赖。
本页说明历史研究，不把这些结果称为当前图片识别、MCP效果或独立盲测准确率。

The original research tried masked reconstruction and supervised band-edge/type
analysis on numerical E(k) tensors. Current scope separates document perception,
deterministic measurements and scientific acceptance instead of presenting an
unverified inverse-structure predictor.

## Data and environment

- AFLOW numerical source: 60,000 selected records; 59,899 usable tensors and 101
  missing required band-edge envelopes. Each input has 128 sampled positions and
  six edge/curvature/distance channels; it is not a complete all-band image.
- Space-group-disjoint outer partition: 47,912 training and 11,987 test samples.
  Separate inner group validation selects checkpoints. Group separation alone is
  not proof of absence of every form of leakage or a validated OOD detector.
- Recorded server: 2×Tesla V100-SXM2-16GB; individual runs were GPU-checked, not
  necessarily distributed across both devices.
- Archived software: Python3.11, TensorFlow2.21, CUDA12.5.82, cuDNN9.3.0.75.
  These are historical records, not Docker dependencies or a newly verified GPU run.

Reference: [AFLOW](https://doi.org/10.1016/j.commatsci.2012.02.005),
[TensorFlow](https://www.usenix.org/conference/osdi16/technical-sessions/presentation/abadi),
[Transformer attention](https://papers.nips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html).

## Programs and method / 训练程序

A **source-inspection snapshot** of the current guarded research entry points and
their local Python import dependencies is included separately:

- [Masked reconstruction entry](../research/historical_training/scripts/train_ssl.py)
- [Supervised entry](../research/historical_training/scripts/finetune_supervised.py)
- [Source hashes](../research/historical_training/source-manifest.json)

这不是“精确复现54轮历史实验”的承诺：现行代码增加了来源合约和验收门槛；历史运行
所需数据、权重和全部环境产物未公开。研究目录不进入MCP镜像，不由MCP工具调用。

Conceptual procedure:

1. Build numerical tensors, record omissions and freeze the group split.
2. Fit normalization on inner training data only.
3. Mask approximately25% of input points and train an attention-based reconstruction
   network; do not mix disconnected path segments or claim physical strain from a k warp.
4. Fine-tune classification and band-edge tasks; select using inner validation.
5. Restore the selected state, bind hashes and evaluate on held-out data afterward.

The actual pretraining entry includes:

```python
trainer.train(train_ds, val_ds, epochs=args.epochs)
```

Model/data/contract construction and GPU checks are in the linked full source, not
omitted instructions for running this line alone. Keep research dependencies in a
separate environment; inspect each CLI's required tensor/selection contracts before
training. Do not disable gates to make historical files look newly accepted.

## What the figure means

![Historical training](figures/historical_training_evaluation.png)

The figure uses the recovered seed42 archive with54 logged epochs and best epoch34.
Its weight digest matches its selection record. A separate active historical directory
contained a51-epoch CSV beside54-epoch metrics; those mismatched records were not merged.

The stored three-class confusion matrix contains11,987 examples. Recomputing its
diagonal/total gives accuracy94.202052%; averaging class F1 gives90.555160%.
[Aggregated plot input and source hashes](figures/paper_metrics.json) are included;
these hashes identify private evidence but do not make it independently accessible.

Limitations: no new GPU training for v2; no matched AI-only versus AI+MCP study;
the1000-item figure corpus is AI-assisted, not independent human ground truth;
calibrated uncertainty is not established. A direct analytic gap baseline exists,
so near-zero learned gap error is not standalone evidence of new physical prediction.
