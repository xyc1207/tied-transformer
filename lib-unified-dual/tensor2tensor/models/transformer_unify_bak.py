# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""transformer (attention).

encoder: [Self-Attention, Feed-forward] x n
decoder: [Self-Attention, Source-Target-Attention, Feed-forward] x n
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

from tensorflow.python.util import nest
from .transformer import Transformer, transformer_prepare_decoder, transformer_encoder

@registry.register_model("transformer_unify")
class Transformer_Unify(Transformer):
  """Attention net.  See file docstring."""

  def omni_encoder(self, inputs_list, target_space_list, hparams, name="encoder"):
    """Encode transformer inputs.

    Args:
      inputs: Transformer inputs [batch_size, input_length, hidden_dim]
      target_space: scalar, target space ID.
      hparams: hyperparmeters for model.

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_decoder_attention_bias: Bias and mask weights for
              encodre-decoder attention. [batch_size, input_length]
    """
    if not isinstance(inputs_list, list):
      inputs_list = [inputs_list]
    if not isinstance(target_space_list, list):
      target_space_list = [target_space_list]

    encoder_output_list, encoder_decoder_attention_bias_list = [], []
    inputs_list = [common_layers.flatten4d3d(x) for x in inputs_list]

    # prepare
    encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
        transformer_prepare_encoder(inputs_list[0], target_space_list[0], hparams))
    encoder_decoder_attention_bias_list.append(encoder_decoder_attention_bias)
    encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.layer_prepostprocess_dropout)
    encoder_output_list.append(transformer_encoder(encoder_input, self_attention_bias, hparams, name))

    for idx in xrange(1, len(inputs_list)):
      encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
          transformer_prepare_encoder(inputs_list[idx], target_space_list[idx], hparams, reuse=True))
      encoder_decoder_attention_bias_list.append(encoder_decoder_attention_bias)
      encoder_input = tf.nn.dropout(encoder_input, 1.0 - hparams.layer_prepostprocess_dropout)
      encoder_output_list.append(transformer_encoder(encoder_input, self_attention_bias, hparams, name, reuse=True))
    return encoder_output_list, encoder_decoder_attention_bias_list

  def model_fn_body(self, features):
    """Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
          "tragets": Target decoder outputs.
              [batch_size, decoder_length, hidden_dim]
          "target_space_id"

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams
    inputs = features.get("inputs")
    targets = features["targets"]
    encoder_name = "omni" if hparams.share_ed else "encoder"
    decoder_name = "omni" if hparams.share_ed else "decoder"

    if inputs is None:
      raise NotImplementedError("Temporally not supported yet")

    encoder_output_list, encoder_decoder_attention_bias_list = self.omni_encoder(
        [inputs, targets], [features["target_space_id"], features["input_space_id"]], hparams, name=encoder_name)

    decoder_input_list = [common_layers.flatten4d3d(targets), common_layers.flatten4d3d(inputs)]
    decoder_output_list = []

    decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(decoder_input_list[0], hparams)
    decoder_output_list.append(
        self.decode(decoder_input, encoder_output_list[0],
                    encoder_decoder_attention_bias_list[0], decoder_self_attention_bias, hparams,
                    name=decoder_name,
                    reuse_common=True if hparams.share_ed else None,
                    reuse_crosslingual_attn=None)
    )

    for idx in xrange(1, len(decoder_input_list)):
      decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(decoder_input_list[idx], hparams)
      decoder_output_list.append(
          self.decode(decoder_input, encoder_output_list[idx],
                      encoder_decoder_attention_bias_list[idx], decoder_self_attention_bias, hparams,
                      name=decoder_name, reuse_common=True, reuse_crosslingual_attn=True)
      )
    return decoder_output_list

  def _greedy_infer(self, features, decode_length):
    """Fast version of greedy decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.

    Returns:
       samples: [batch_size, input_length + decode_length]
       logits: Not returned
       losses: Not returned

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    decoded_ids, _ = self._fast_decode(features, decode_length)
    return decoded_ids, None, None

  def _beam_decode(self, features, decode_length, beam_size, top_beams,
                   alpha):
    """Beam search decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.

    Returns:
       samples: an integer `Tensor`. Top samples from the beam search
    """
    decoded_ids, scores = self._fast_decode(
        features, decode_length, beam_size, top_beams, alpha)
    return {"outputs": decoded_ids, "scores": scores}

  def _fast_decode(self,
                   features,
                   decode_length,
                   beam_size=1,
                   top_beams=1,
                   alpha=1.0):
    """Fast decoding.

    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      features: a map of string to model  features.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.

    Returns:
       samples: an integer `Tensor`. Top samples from the beam search

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
      raise NotImplementedError("Fast decoding only supports a single shard.")
    dp = self._data_parallelism
    hparams = self._hparams

    encoder_name = "omni" if hparams.share_ed else "encoder"
    decoder_name = "omni" if hparams.share_ed else "decoder"

    inputs = features["inputs"]
    batch_size = tf.shape(inputs)[0]
    target_modality = self._problem_hparams.target_modality
    if t2t_model.is_class_modality(target_modality):
      decode_length = 1
    else:
      decode_length = tf.shape(inputs)[1] + decode_length

    # TODO(llion): Clean up this reshaping logic.
    inputs = tf.expand_dims(inputs, axis=1)
    if len(inputs.shape) < 5:
      inputs = tf.expand_dims(inputs, axis=4)
    s = tf.shape(inputs)
    inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
    # _shard_features called to ensure that the variable names match
    inputs = self._shard_features({"inputs": inputs})["inputs"]
    input_modality = self._problem_hparams.input_modality["inputs"]
    with tf.variable_scope(input_modality.name):
      inputs = input_modality.bottom_sharded(inputs, dp)

    with tf.variable_scope("body"):
      encoder_output, encoder_decoder_attention_bias = dp(
          self.omni_encoder, inputs, features["target_space_id"], hparams, encoder_name)
    encoder_output = encoder_output[0][0]
    encoder_decoder_attention_bias = encoder_decoder_attention_bias[0][0]

    if hparams.pos == "timing":
      timing_signal = common_attention.get_timing_signal_1d(
          decode_length + 1, hparams.hidden_size)

    def preprocess_targets(targets, i):
      """Performs preprocessing steps on the targets to prepare for the decoder.

      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.

      Args:
        targets: inputs ids to the decoder. [batch_size, 1]
        i: scalar, Step number of the decoding loop.

      Returns:
        Processed targets [batch_size, 1, hidden_dim]
      """
      # _shard_features called to ensure that the variable names match
      targets = self._shard_features({"targets": targets})["targets"]
      with tf.variable_scope(target_modality.name):
        targets = target_modality.targets_bottom_sharded(targets, dp)[0]
      targets = common_layers.flatten4d3d(targets)

      # TODO(llion): Explain! Is this even needed?
      targets = tf.cond(
          tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

      if hparams.pos == "timing":
        targets += timing_signal[:, i:i + 1]
      return targets

    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
      decoder_self_attention_bias += common_attention.attention_bias_proximal(
          decode_length)

    def symbols_to_logits_fn(ids, i, cache):
      """Go from ids to logits for next symbol."""
      ids = ids[:, -1:]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = preprocess_targets(targets, i)

      bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      with tf.variable_scope("body"):
        body_outputs = dp(self.decode, targets, cache["encoder_output"],
                          cache["encoder_decoder_attention_bias"], bias,
                          hparams, cache, decoder_name, True if hparams.share_ed else None, None)

      with tf.variable_scope(target_modality.name):
        logits = target_modality.top_sharded(body_outputs, None, dp)[0]

      return tf.squeeze(logits, axis=[1, 2, 3]), cache

    key_channels = hparams.attention_key_channels or hparams.hidden_size
    value_channels = hparams.attention_value_channels or hparams.hidden_size
    num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers

    cache = {
        "layer_%d" % layer: {
            "k": tf.zeros([batch_size, 0, key_channels]),
            "v": tf.zeros([batch_size, 0, value_channels]),
        }
        for layer in range(num_layers)
    }

    # Set 2nd dim to None since it's not invariant in the tf.while_loop
    # Note: Tensor.set_shape() does not work here since it merges shape info.
    # TODO(llion); Find a more robust solution.
    # pylint: disable=protected-access
    for layer in cache:
      cache[layer]["k"]._shape = tf.TensorShape([None, None, key_channels])
      cache[layer]["v"]._shape = tf.TensorShape([None, None, value_channels])
    # pylint: enable=protected-access
    cache["encoder_output"] = encoder_output
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

    if beam_size > 1:  # Beam Search
      target_modality = (
          self._hparams.problems[self._problem_idx].target_modality)
      vocab_size = target_modality.top_dimensionality
      initial_ids = tf.zeros([batch_size], dtype=tf.int32)
      decoded_ids, scores = beam_search.beam_search(
          symbols_to_logits_fn, initial_ids, beam_size, decode_length,
          vocab_size, alpha, states=cache, stop_early=(top_beams == 1))

      if top_beams == 1:
        decoded_ids = decoded_ids[:, 0, 1:]
      else:
        decoded_ids = decoded_ids[:, :top_beams, 1:]
    else:  # Greedy

      def inner_loop(i, next_id, decoded_ids, cache):
        logits, cache = symbols_to_logits_fn(next_id, i, cache)
        temperature = (0.0 if hparams.sampling_method == "argmax"
                       else hparams.sampling_temp)
        next_id = tf.expand_dims(
            common_layers.sample_with_temperature(logits, temperature), axis=1)
        decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
        return i + 1, next_id, decoded_ids, cache

      decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int64)
      scores = None
      next_id = tf.zeros([batch_size, 1], dtype=tf.int64)
      _, _, decoded_ids, _ = tf.while_loop(
          # TODO(llion): Early stopping.
          lambda i, *_: tf.less(i, decode_length),
          inner_loop,
          [tf.constant(0), next_id, decoded_ids, cache],
          shape_invariants=[
              tf.TensorShape([]),
              tf.TensorShape([None, None]),
              tf.TensorShape([None, None]),
              nest.map_structure(lambda t: tf.TensorShape(t.shape), cache),
          ])

    return decoded_ids, scores



def transformer_prepare_encoder(inputs, target_space, hparams, reuse=None):
  """Prepare one shard of the model for the encoder.

  Args:
    inputs: a Tensor.
    target_space: a Tensor.
    hparams: run hyperparameters

  Returns:
    encoder_input: a Tensor, bottom of encoder stack
    encoder_self_attention_bias: a bias tensor for use in encoder self-attention
    encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
      attention
  """
  ishape_static = inputs.shape.as_list()
  encoder_input = inputs
  encoder_padding = common_attention.embedding_to_padding(encoder_input)
  ignore_padding = common_attention.attention_bias_ignore_padding(
      encoder_padding)
  encoder_self_attention_bias = ignore_padding
  encoder_decoder_attention_bias = ignore_padding
  if hparams.proximity_bias:
    encoder_self_attention_bias += common_attention.attention_bias_proximal(
        tf.shape(inputs)[1])
  # Append target_space_id embedding to inputs.
  emb_target_space = common_layers.embedding(
      target_space, 32, ishape_static[-1], name="target_space_embedding", reuse=reuse)
  emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
  encoder_input += emb_target_space
  if hparams.pos == "timing":
    encoder_input = common_attention.add_timing_signal_1d(encoder_input)
  return (encoder_input, encoder_self_attention_bias,
          encoder_decoder_attention_bias)



