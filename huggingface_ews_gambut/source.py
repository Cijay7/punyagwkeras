"""Client REST sumber HTTPS; URL hanya berasal dari konfigurasi server."""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from pydantic import ValidationError

from core import Devices, History


class SourceError(Exception):
    def __init__(self, code, message, status=503):
        self.code, self.message, self.status = code, message, status


class SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old, new = urllib.parse.urlsplit(req.full_url), urllib.parse.urlsplit(newurl)
        if new.scheme != "https" or (new.hostname, new.port or 443) != (old.hostname, old.port or 443):
            raise SourceError("source_redirect_rejected", "Redirect sumber harus tetap pada host HTTPS yang sama.", 502)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Source:
    def __init__(self, base_url="", token="", example=False, example_path=None):
        self.base_url, self.token, self.example = base_url.rstrip("/"), token, example
        self.example_path = Path(example_path) if example_path else None
        if base_url:
            parsed = urllib.parse.urlsplit(base_url)
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or
                    parsed.query or parsed.fragment or parsed.port not in (None, 443, 8080)):
                raise ValueError("SOURCE_API_BASE_URL harus HTTPS port 443/8080 tanpa kredensial/query.")

    @property
    def configured(self):
        return self.example or bool(self.base_url)

    def _get(self, path):
        if not self.base_url:
            raise SourceError("source_not_configured", "Isi Secret SOURCE_API_BASE_URL dan SOURCE_API_TOKEN pada Space.")
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = urllib.request.Request(self.base_url+path, headers=headers)
        try:
            with urllib.request.build_opener(SameOriginRedirect()).open(request, timeout=15) as response:
                body = response.read(2_000_001)
            if len(body) > 2_000_000:
                raise SourceError("source_response_too_large", "Respons sumber melebihi 2 MB.", 502)
            return json.loads(body)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise SourceError("device_not_found", "Device tidak ditemukan pada API sumber.", 404) from None
            raise SourceError("source_http_error", "API sumber menolak atau gagal memproses permintaan.", 502) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise SourceError("source_unavailable", "API sumber HTTPS tidak dapat dihubungi.") from None
        except (ValueError, UnicodeError):
            raise SourceError("source_invalid_json", "Respons API sumber bukan JSON yang sesuai.", 502) from None

    def devices(self):
        if self.example:
            history = History.model_validate_json(self.example_path.read_text(encoding="utf-8"))
            return {"devices": [{"device_id": history.device_id, "latest_time": max(r.date_time for r in history.readings).isoformat()}]}
        try:
            return Devices.model_validate(self._get("/devices")).model_dump(mode="json")
        except ValidationError:
            raise SourceError("source_invalid_data", "Format daftar device dari sumber tidak sesuai kontrak.", 502) from None

    def history(self, device_id):
        if self.example:
            history = History.model_validate_json(self.example_path.read_text(encoding="utf-8"))
            if history.device_id != device_id:
                raise SourceError("device_not_found", "Mode example hanya menyediakan device contoh.", 404)
            return history
        payload = self._get("/devices/"+urllib.parse.quote(device_id, safe="")+"/history?hours=25")
        try:
            history = History.model_validate(payload)
        except ValidationError:
            raise SourceError("source_invalid_data", "Histori sumber harus berisi device_id, time_is_synthetic, dan readings sesuai schema.", 502) from None
        if history.device_id != device_id:
            raise SourceError("source_device_mismatch", "Device pada respons sumber tidak sesuai permintaan.", 502)
        return history
