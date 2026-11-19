# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Publish the real prompt embeddings for the thinker's ``layers[0]`` capture.

``stage_input_processors/qwen3_omni.py`` documents the thinker's captured
layer ``0`` as the *word embedding* (``_EMBED_LAYER_KEY = "0"``) and the talker
conditions its text stream on it::

    assistant_hidden = self.talker.text_projection(thinker_embed[i0:i1])

``Qwen3MoeLLMModel.forward`` captures ``layers[0] = hidden_states.clone()``
right before the first decoder layer, i.e. its input embedding.  With the
thinker stage running under CUDA graphs, that clone is *not* refreshed per
replay: the captured/traced dict returns the tensor as it was during the
capture pass instead of the current step's embeddings.  Measured on C500 with
``VLLM_OMNI_METAX_DEBUG_THINKER_CAPTURE=1``:

* stage 0 under CUDA graphs -> ``layers[0]`` has std 1.75, norms 27..150 and
  no cosine relation to ``embed_tokens(ids)`` at any row shift;
* the same run with ``enforce_eager: true`` -> ``layers[0]`` equals the
  merged prompt embeddings (std 0.30, ``cos == 1.0`` against the runner's
  ``_prepare_mm_inputs`` buffer).

The garbled conditioning explains the whole regression: the thinker's text is
unaffected (it never reads the capture), the Code2Wav decoder is fine (it is
bit-identical to the HF reference implementation on the same codes), yet the
talker turns the wrong ``text_projection`` input into plausible-but-wrong
codec codes, so the audio is speech-like but unintelligible.

This patch keeps the CUDA-graph path and snapshots ``inputs_embeds`` *before*
the model runs, replacing the stale ``layers[0]`` entry in the returned
capture dict with that snapshot.  ``inputs_embeds`` is only valid for the
duration of the call (the runner reuses the buffer), so the snapshot is a
clone; it is one clone of [#tokens, hidden] per forward, the same cost the
in-graph capture was supposed to pay.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def _capture_layer_indices(value) -> set[int]:
    if value is None:
        return set()
    try:
        return {int(v) for v in value}
    except (TypeError, ValueError):
        return set()


def apply_qwen3_omni_thinker_capture_patch() -> None:
    """Replace the stale thinker ``layers[0]`` capture with the real embeddings.

    Purpose: under CUDA graphs the thinker's layer-0 capture (the talker's
    "word embedding" conditioning) is published from the capture pass instead
    of the current step, which garbles Qwen3-Omni audio while the text answer
    stays correct.

    Added: V0.28.0.
    remove_at: the installed vllm-omni publishes a correct ``layers[0]``
    capture with the Qwen3-Omni thinker stage running under CUDA graphs
    (verify with ``VLLM_OMNI_METAX_DEBUG_TALKER_PAYLOAD=1``: ``embed.prefill``
    must have std ~0.3, not ~1.7).
    """
    global _PATCHED

    if _PATCHED:
        return

    try:
        from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )
    except ImportError:
        logger.debug("MetaX: Qwen3-Omni thinker not importable; skipping capture patch.")
        return

    original = Qwen3OmniMoeThinkerForConditionalGeneration.forward
    if getattr(original, "_metax_capture_patched", False):
        _PATCHED = True
        return

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        capture_layer_indices=None,
        return_hidden_states=False,
        **kwargs,
    ):
        snapshot = None
        wants_layer0 = return_hidden_states and 0 in _capture_layer_indices(capture_layer_indices)
        if wants_layer0:
            try:
                import torch

                if isinstance(inputs_embeds, torch.Tensor):
                    # Snapshot while the runner's buffer still holds this
                    # step's embeddings: the buffer is reused, and the
                    # in-graph clone returns the capture pass's contents.
                    snapshot = inputs_embeds.detach().clone()
            except Exception:
                logger.warning(
                    "MetaX: could not snapshot thinker inputs_embeds.", exc_info=True
                )

        out = original(
            self,
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            capture_layer_indices=capture_layer_indices,
            return_hidden_states=return_hidden_states,
            **kwargs,
        )

        if snapshot is not None:
            try:
                captured = out[1] if isinstance(out, tuple) and len(out) > 1 else None
                layers = captured.get("hidden_states", {}).get("layers") if isinstance(captured, dict) else None
                if isinstance(layers, dict) and 0 in layers:
                    layers[0] = snapshot.reshape(-1, snapshot.shape[-1])
            except Exception:
                logger.warning(
                    "MetaX: could not restore the thinker layer-0 capture.", exc_info=True
                )
        return out

    forward._metax_capture_patched = True
    Qwen3OmniMoeThinkerForConditionalGeneration.forward = forward
    _PATCHED = True
    logger.warning(
        "MetaX: patched the Qwen3-Omni thinker layer-0 capture to publish the "
        "current step's prompt embeddings (fixes unintelligible audio)."
    )
