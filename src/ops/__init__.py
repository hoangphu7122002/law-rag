"""Where code runs. Science code never imports this package; only cli.py does.

RemoteBox     a GPU machine over ssh: sync code / data, start vLLM in tmux, run experiments, pull results
modal_app.py  Modal jobs (`modal run src/ops/modal_app.py::<job>`), kept for the Modal credit
"""
from ops.box import RemoteBox

__all__ = ["RemoteBox"]
