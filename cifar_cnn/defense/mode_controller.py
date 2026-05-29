# REMOVED: replaced by R_i weighted aggregation


class ModeController:
    """Stub — ModeController removed. R_i weighted aggregation is always used."""

    def __init__(self, **kwargs):
        pass

    def update_mode(self, *args, **kwargs) -> str:
        return "NORMAL"

    def get_current_mode(self) -> str:
        return "NORMAL"

    def get_aggregation_algorithm(self) -> str:
        return "weighted_average"

    def get_stats(self):
        return {}

    def get_config(self):
        return {}
