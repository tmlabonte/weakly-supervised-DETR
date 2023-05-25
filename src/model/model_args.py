"""Define model configuration arguments."""

def add_model_args(parent_parser):
    """Adds model configuration arguments to parser."""

    parser = parent_parser.add_argument_group("DeformableDETR")

    parser.add(
        "--activation",
        choices=["relu", "gelu", "glu"],
        type=str,
        help="Which activation to use in the transformer.",
    )
    parser.add(
        "--dec_layers",
        type=int,
        help="How many layers to use in the decoder.",
    )
    parser.add(
        "--dec_points",
        type=int,
        help="How many reference points to use in the decoder.",
    )
    parser.add(
        "--dilation",
        action="store_true",
        help="Whether to replace stride with dilation in the last conv block.",
    )
    parser.add(
        "--enc_layers",
        type=int,
        help="How many layers to use in the encoder.",
    )
    parser.add(
        "--enc_points",
        type=int,
        help="How many reference points to use in the encoder.",
    )
    parser.add(
        "--feature_levels",
        type=int,
        help="How many feature levels to use in the Transformer.",
    )
    parser.add(
        "--feedforward_dim",
        type=int,
        help="Intermediate size of the feedforward layers in the Transformer.",
    )
    parser.add(
        "--heads",
        type=int,
        help="How many attention heads to use in the Transformer.",
    )
    parser.add(
        "--hidden_dim",
        type=int,
        help="Transformer embedding dimensionality.",
    )
    parser.add(
        "--position_embedding",
        choices=["learned", "sine"],
        help="Type of positional embedding to use on the image features.",
    )
    parser.add(
        "--position_embedding_scale",
        type=float,
        help="Scale of the transformer position embedding.",
    )
    parser.add(
        "--queries",
        type=int,
        help="How many object queries to use in the transformer.",
    )

    return parent_parser
  
