"""Install the shared in-process Calendar service for a test."""
from fake_calendar import (ACM, HOLIDAYS, IMPORTED, MILES, NSBE, PRIMARY,
                           FakeCalendar, Service, _Exec, http_error)


def install(monkeypatch, fake=None):
    """Make calendar_tools talk to a FakeCalendar. Returns it."""
    import calendar_tools
    fake = fake or FakeCalendar()
    calendar_tools._calendar_cache.clear()
    monkeypatch.setattr(calendar_tools, "_service", lambda: Service(fake))
    return fake
