"""Weather for the home screen greeting ("Hot & sunny Monday, 94°F in Austin").

Location comes from the visitor's IP address (ipwho.is, free, no key), so there's no permission prompt. Weather
comes from Open-Meteo (free, no key). Both are cached in memory: locations for an hour per address, weather for
15 minutes per rounded position. Nothing is logged or stored. If a lookup fails, or the address is private
(local testing), the greeting falls back to Austin, where Harsha is.
"""
from __future__ import annotations

import ipaddress
import json
import threading
import time
import urllib.parse
import urllib.request
from typing import Callable

FALLBACK = {"city": "Austin", "region": "Texas", "country": "US", "lat": 30.27, "lon": -97.74, "fallback": True}
FAHRENHEIT_COUNTRIES = {"US", "LR", "MM", "BS", "BZ", "KY", "PW", "FM", "MH"}


def classify(code: int, feels_c: float, is_day: bool) -> str:
    """WMO weather code + temperature -> one of: storm, snow, rain, fog, hot, cold, sunny, clear, cloudy."""
    if code >= 95:
        return "storm"
    if 71 <= code <= 77 or code in (85, 86):
        return "snow"
    if 51 <= code <= 67 or 80 <= code <= 82:
        return "rain"
    if code in (45, 48):
        return "fog"
    if feels_c >= 32:
        return "hot"
    if feels_c <= 3:
        return "cold"
    if code <= 1:
        return "sunny" if is_day else "clear"
    return "cloudy"


def _default_fetch(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "ask-harsha"})
    with urllib.request.urlopen(req, timeout=4) as r:
        return json.loads(r.read().decode())


class Weather:
    def __init__(self, fetch: Callable[[str], dict] | None = None, clock=time.monotonic):
        self.fetch = fetch or _default_fetch
        self.clock = clock
        self._places: dict[str, tuple[float, dict]] = {}
        self._weather: dict[tuple, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def _cached(self, store: dict, key, ttl: float, make):
        now = self.clock()
        with self._lock:
            hit = store.get(key)
            if hit and now - hit[0] < ttl:
                return hit[1]
        value = make()
        with self._lock:
            if len(store) > 5000:
                store.clear()
            store[key] = (now, value)
        return value

    def locate(self, ip: str) -> dict:
        try:
            addr = ipaddress.ip_address(ip)
            if not addr.is_global:
                return FALLBACK
        except ValueError:
            return FALLBACK

        def make():
            try:
                d = self.fetch(f"https://ipwho.is/{urllib.parse.quote(ip)}?fields=success,city,region,country_code,latitude,longitude")
                if d.get("success") and d.get("latitude") is not None:
                    return {"city": d.get("city") or "", "region": d.get("region") or "", "country": d.get("country_code") or "",
                            "lat": float(d["latitude"]), "lon": float(d["longitude"]), "fallback": False}
            except Exception:
                pass
            return FALLBACK
        return self._cached(self._places, ip, 3600, make)

    def locate_city(self, name: str) -> dict | None:
        """A place someone typed in Customize ("Chicago", "Hyderabad"), via Open-Meteo's free geocoder."""
        name = " ".join(name.split())[:80]
        if not name:
            return None

        def make():
            try:
                d = self.fetch("https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
                    {"name": name.split(",")[0], "count": 5, "language": "en", "format": "json"}))
                results = d.get("results") or []
                hint = name.split(",", 1)[1].strip().lower() if "," in name else ""
                if hint:  # "Austin, TX" or "Paris, France": prefer the result whose region or country matches
                    results = sorted(results, key=lambda r: not any(hint in str(r.get(k, "")).lower() or
                                                                    str(r.get(k, "")).lower().startswith(hint)
                                                                    for k in ("admin1", "country", "country_code")))
                if results:
                    r = results[0]
                    return {"city": r.get("name") or name, "region": r.get("admin1") or "", "country": r.get("country_code") or "",
                            "lat": float(r["latitude"]), "lon": float(r["longitude"]), "fallback": False}
            except Exception:
                pass
            return None
        return self._cached(self._places, "city:" + name.lower(), 86400, make)

    def current(self, lat: float, lon: float) -> dict | None:
        key = (round(lat, 1), round(lon, 1))

        def make():
            try:
                d = self.fetch("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
                    "latitude": key[0], "longitude": key[1],
                    "current": "temperature_2m,apparent_temperature,weather_code,is_day", "timezone": "auto"}))
                c = d["current"]
                return {"temp_c": float(c["temperature_2m"]), "feels_c": float(c.get("apparent_temperature", c["temperature_2m"])),
                        "code": int(c["weather_code"]), "is_day": bool(c.get("is_day", 1))}
            except Exception:
                return None
        return self._cached(self._weather, key, 900, make)

    def greeting(self, ip: str, city: str = "") -> dict:
        place = (self.locate_city(city) if city else None) or self.locate(ip)
        now = self.current(place["lat"], place["lon"])
        if now is None:
            return {"ok": False, "city": place["city"]}
        fahrenheit = place["country"] in FAHRENHEIT_COUNTRIES
        conv = (lambda c: round(c * 9 / 5 + 32)) if fahrenheit else (lambda c: round(c))
        return {"ok": True, "city": place["city"], "region": place["region"], "fallback": place["fallback"],
                "condition": classify(now["code"], now["feels_c"], now["is_day"]), "is_day": now["is_day"],
                "temp": conv(now["temp_c"]), "feels": conv(now["feels_c"]), "unit": "F" if fahrenheit else "C"}
