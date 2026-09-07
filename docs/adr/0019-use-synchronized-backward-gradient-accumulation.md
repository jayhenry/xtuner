# Require synchronized-backward gradient accumulation

XTuner requires its current accumulation behavior for all FSDP model paths:
every accumulation iteration calls `loss.backward()` and executes FSDP
ReduceScatter. MoonEP completes duplicated BF16 expert gradients before that
reduction. Accumulation across iterations
therefore happens only in FSDP-owned FP32 sharded gradients, after the MoonEP
workspace has been consumed. Each XTuner FSDP unit rejects
`set_requires_gradient_sync(False)` immediately, whether called on the root or
a child. `set_is_last_backward(False)` remains available for FSDP backward-state
management; it does not disable ReduceScatter. Native PyTorch modules outside
XTuner retain their original behavior.

Keeping full BF16 gradients for all layers across accumulation iterations is
outside XTuner's memory policy. Activation recomputation does not remove that
storage cost. MoonEP additionally hands each completed home-gradient view to
FSDP by assignment followed by in-place addition, so even Domino micro2 retains
a reusable VMM slot alias. Delaying RS would allow other layers or subsequent
backwards to overwrite that gradient.

The supported contract is therefore synchronous gradient reduction on every
backward, with accumulation only in FSDP-owned FP32 shards.
