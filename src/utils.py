def format_duration(seconds: float) -> str:
    """Converte i secondi in un formato leggibile (giorni, ore, minuti)."""
    if seconds < 60:
        return f"{seconds:.2f} secondi"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f} minuti"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.2f} ore"
    days = hours / 24
    if days < 365:
        return f"{days:.2f} giorni"
    years = days / 365
    return f"{years:.2f} anni"