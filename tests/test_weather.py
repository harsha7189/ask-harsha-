import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from agent import inbox as inboxmod, receipts as rcpt  # noqa: E402
from agent.notes import MemoryNotesStore  # noqa: E402
from agent.weather import FALLBACK, Weather, classify  # noqa: E402
from fakes import FakeClient  # noqa: E402
from server.app import Settings, create_app  # noqa: E402


def fake_fetch(calls, code=0, temp=35.0, feels=38.0, is_day=1, geo=None, fail_weather=False):
    def fetch(url):
        calls.append(url)
        if "ipwho.is" in url:
            return geo or {"success": True, "city": "Austin", "region": "Texas", "country_code": "US", "latitude": 30.27, "longitude": -97.74}
        if fail_weather:
            raise OSError("down")
        return {"current": {"temperature_2m": temp, "apparent_temperature": feels, "weather_code": code, "is_day": is_day}}
    return fetch


@pytest.mark.parametrize("code, feels, day, expected", [
    (0, 38, True, "hot"), (0, 22, True, "sunny"), (0, 22, False, "clear"), (3, 22, True, "cloudy"),
    (61, 15, True, "rain"), (81, 15, True, "rain"), (95, 25, True, "storm"), (73, -3, True, "snow"),
    (45, 10, True, "fog"), (2, 0, True, "cold"),
])
def test_classify(code, feels, day, expected):
    assert classify(code, feels, day) == expected


def test_us_visitors_get_fahrenheit_and_lookups_are_cached():
    calls = []
    w = Weather(fake_fetch(calls))
    g = w.greeting("8.8.8.8")
    assert g == {"ok": True, "city": "Austin", "region": "Texas", "fallback": False, "condition": "hot", "is_day": True,
                 "temp": 95, "feels": 100, "unit": "F"}
    w.greeting("8.8.8.8")
    assert len(calls) == 2  # one location lookup, one weather lookup


def test_other_countries_get_celsius():
    w = Weather(fake_fetch([], code=61, temp=14.4, feels=12.6, geo={"success": True, "city": "Hyderabad", "country_code": "IN", "latitude": 17.4, "longitude": 78.5}))
    g = w.greeting("1.2.3.4")
    assert g["unit"] == "C" and g["temp"] == 14 and g["condition"] == "rain" and g["city"] == "Hyderabad"


def test_private_address_falls_back_to_austin_without_a_lookup():
    calls = []
    g = Weather(fake_fetch(calls)).greeting("127.0.0.1")
    assert g["city"] == FALLBACK["city"] and g["fallback"] is True and not any("ipwho" in c for c in calls)


def test_weather_outage_is_not_an_error():
    assert Weather(fake_fetch([], fail_weather=True)).greeting("8.8.8.8") == {"ok": False, "city": "Austin"}


def test_greeting_endpoint():
    s = Settings(daily_budget_usd=5.0, rate_limit=50, allowed_origins=[], owner_passcode="open-sesame")
    app = create_app(client=FakeClient([]), settings=s, notes=MemoryNotesStore(), inbox=inboxmod.MemoryInboxStore(),
                     receipts=rcpt.ReceiptStore(inboxmod.make_key("k")), weather=Weather(fake_fetch([], code=71, temp=-2, feels=-6)))
    c = TestClient(app)
    d = c.get("/api/greeting", headers={"x-forwarded-for": "8.8.8.8"}).json()
    assert d["condition"] == "snow" and d["unit"] == "F" and d["temp"] == 28
