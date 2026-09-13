class PaisaError(Exception):
    """Base exception for PAISA-specific errors."""


class DataSourceError(PaisaError):
    """Raised when a data provider cannot fetch or parse data."""


class DatasetNotReadyError(PaisaError):
    """Raised when generated data/model artifacts are missing."""
