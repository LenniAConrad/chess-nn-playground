# Ablations

i249 is an execution rewrite of i018, not a new learning hypothesis. Architecture
ablations remain i018 ablations; i249-specific ablations only test execution
choices and numerical faithfulness.

## Execution Ablations

| ID | Switch | What it tests |
|---|---|---|
| F1 | `model.compile_model: false` | Algebraic block speed without `torch.compile`. |
| F2 | `model.compile_mode: default` | Kernel fusion without the reduce-overhead/CUDA-graph path. |
| F3 | `model.compile_mode: max-autotune` | Whether autotuned kernels beat reduce-overhead on the host GPU. |
| F4 | `model.return_diagnostics: false` | Serving-only logits path; logits must match full-output mode. |
| F5 | `model.inference_autocast_dtype: none / float16 / bfloat16` | Precision/speed tradeoff for eval and serving. Exact-equivalence audits should use `none`; fastest local path was `float16`. |

The two maintained comparison configs are:

- `config.yaml`: exact i249 speedup (`inference_autocast_dtype: none`).
- `config_eval_fp16.yaml`: lower-precision eval speedup
  (`inference_autocast_dtype: float16`).

## Keep / Drop Rule

Keep i249 only if all are true:

- shared-weight numerical check passes (`logits <= 1e-5`, checked gradients
  `<= 1e-7`);
- matched paper-grade PR-AUC remains within i018's expected seed noise;
- same-GPU inference throughput is clearly above i018;
- no unexpected slice regression appears in the inherited CRTK reports.

Drop i249 if the speedup depends on changing inputs, labels, loss, reporting
metadata, or benchmark accounting. The only allowed difference is execution
order.
