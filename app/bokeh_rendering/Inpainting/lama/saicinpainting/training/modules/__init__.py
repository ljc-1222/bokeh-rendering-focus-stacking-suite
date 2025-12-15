import logging

def make_generator(config, kind, **kwargs):
    """Construct the LaMa generator module.

    Note:
        This `bokeh_rendering_and_focus_stacking_suite/` folder is trimmed for *inference only*.
        The shipped `big-lama/config.yaml` uses `generator.kind: ffc_resnet`,
        so we keep only that path and avoid importing other (unused) training
        architectures at module import time.
    """
    logging.info("Make generator %s", kind)

    if kind == "ffc_resnet":
        # Lazy import: keeps the inference dependency surface small.
        from saicinpainting.training.modules.ffc import FFCResNetGenerator

        return FFCResNetGenerator(**kwargs)

    raise ValueError(
        f"Unknown generator kind {kind!r}. "
        "This trimmed Dr.Bokeh build only supports 'ffc_resnet' for inference."
    )


def make_discriminator(kind, **kwargs):
    """Discriminator factory (not included in inference-only build)."""
    raise ValueError(
        f"Discriminator kind {kind!r} requested, but this trimmed build is inference-only."
    )
