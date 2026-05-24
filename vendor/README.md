# Vendor Namespace

`vendor/` is reserved as the canonical namespace for third-party source trees.

The current compatibility implementation still loads the existing vendored
DPT, LaMa, and LDF code from `app/bokeh_rendering/`. New code should not add new
hardcoded references to those legacy paths; use `brnfs.paths` instead.
