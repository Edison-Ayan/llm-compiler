# Milestone 1: frontend contract

## Decision

The compiler frontend consumes `torch.export.ExportedProgram`, not arbitrary
Python modules and not pretty-printed FX source.

This gives the next stage:

- a functional ATen graph;
- lifted parameters with explicit graph-signature roles;
- symbolic dimensions and range constraints;
- tensor metadata produced during export;
- a serializable artifact (`.pt2`) that can be tested independently.

The JSON summary is an inspection and regression artifact. It is not the
compiler's authoritative IR and should not be executed.

## Input invariants

The first frontend fixture accepts:

```text
hidden_states  : f32[batch, sequence, hidden]
attention_mask : f32[batch, 1, sequence, sequence]
```

Constraints:

```text
1 <= batch <= max_batch
1 <= sequence <= max_sequence
hidden = compile-time constant
attention_mask.dim(0) == hidden_states.dim(0)
attention_mask.dim(2) == hidden_states.dim(1)
attention_mask.dim(3) == hidden_states.dim(1)
```

Batch and sequence are runtime symbolic values. Model dimensions such as
hidden size, head count and head dimension are compile-time configuration.

## Correctness contract

For every input satisfying the guards, the exported program must:

1. return the same shape as `hidden_states`;
2. match eager execution within PyTorch's default floating-point tolerance;
3. survive save/load without losing dynamic-shape support.

Tests exercise both the capture shape and a different runtime shape.

## Boundary of the next importer

The ServeIR importer will initially recognize only a deliberately small ATen
subset:

```text
linear, pow, mean, rsqrt, add, mul, silu
view, transpose, split, getitem, repeat_interleave
matmul, softmax, sym_size
```

Unsupported operators remain explicit external calls. This avoids making the
first importer depend on a complete ATen lowering.

## Stateful Decode extension

The original frontend remains a pure Prefill-style fixture. The Stateful Decode
extension now exports tensor-form past key/value inputs and rewrites them into:

```text
!serve.kv_state
serve.kv.read
serve.kv.append
```

These operations carry explicit read/write effects. See
`docs/stateful-decode.md` for the dynamic cache-length contract, state rewrite,
runtime semantics and current limitations. Paged allocation and prefix-cache
behavior remain future work.
