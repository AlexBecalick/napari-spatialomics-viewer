"""Shared display-only controls for saved bootstrap predictions."""

from qtpy.QtWidgets import QDoubleSpinBox, QLabel, QVBoxLayout, QWidget


class BootstrapFilterWidget(QWidget):
    """An inclusive minimum support, expressed as a percentage; zero shows all."""

    def __init__(self, callback=None):
        super().__init__()
        self._callback = callback
        self.cutoff = QDoubleSpinBox()
        self.cutoff.setRange(0, 100)
        self.cutoff.setDecimals(1)
        self.cutoff.setSingleStep(5)
        self.cutoff.setSuffix(" %")
        self.cutoff.setValue(0)
        self.cutoff.valueChanged.connect(self._changed)
        note = QLabel(
            "Bootstrap support measures stability, not biological probability. "
            "0% shows all predictions. Below-cut-off cells are grey on UMAP "
            "and unfilled in the spatial view. Accepted annotations stay separate."
        )
        note.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Minimum bootstrap support"))
        layout.addWidget(self.cutoff)
        layout.addWidget(note)
        self.setEnabled(False)

    def _changed(self, percent):
        if self._callback is not None:
            self._callback(float(percent) / 100)

    def set_filter(self, value, enabled):
        self.cutoff.blockSignals(True)
        self.cutoff.setValue(float(value) * 100)
        self.cutoff.blockSignals(False)
        self.setEnabled(bool(enabled))
