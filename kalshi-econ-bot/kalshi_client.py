"""Thin wrapper around the Kalshi trade API v2.

Kalshi authenticates requests with an API key ID plus an RSA-PSS signature
over ``timestamp + method + path``. Private keys are loaded from disk and
never transmitted. This client targets the demo environment by default and
exposes only the handful of endpoints the bot uses.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

import config
from logger import get_logger

log = get_logger(__name__)


class KalshiAPIError(Exception):
    """Raised when the Kalshi API returns an unrecoverable error."""


class KalshiClient:
    """Minimal Kalshi v2 client with request signing and backoff."""

    def __init__(
        self,
        api_key_id: str | None = None,
        private_key_path: str | None = None,
        private_key_password: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key_id = api_key_id or config.KALSHI_API_KEY
        self.base_url = (base_url or config.KALSHI_BASE_URL).rstrip("/")
        self._private_key: rsa.RSAPrivateKey | None = None

        key_path = Path(private_key_path or config.KALSHI_PRIVATE_KEY_PATH)
        pw = (private_key_password or config.KALSHI_PRIVATE_KEY_PASSWORD)
        if self.api_key_id and key_path.exists():
            self._private_key = self._load_private_key(key_path, pw)
        else:
            log.warning(
                "Kalshi credentials missing or private key not found at %s; "
                "client will only work for public endpoints.", key_path,
            )

        self.session = requests.Session()

    # ------------------------------------------------------------------ auth

    @staticmethod
    def _load_private_key(
        path: Path, password: str | None
    ) -> rsa.RSAPrivateKey:
        """Load an RSA private key in PEM format from ``path``."""
        data = path.read_bytes()
        key = serialization.load_pem_private_key(
            data, password=password.encode() if password else None
        )
        if not isinstance(key, rsa.RSAPrivateKey):
            raise KalshiAPIError("Kalshi private key must be an RSA key.")
        return key

    def _sign(self, timestamp_ms: str, method: str, path: str) -> str:
        """Return base64 RSA-PSS signature over ``timestamp + method + path``.

        Kalshi's docs specify the path portion includes ``/trade-api/v2`` but
        excludes query params. We sign exactly what Kalshi expects.
        """
        if self._private_key is None:
            raise KalshiAPIError("Private key not loaded; cannot sign request.")
        message = f"{timestamp_ms}{method.upper()}{path}".encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        timestamp_ms = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": self._sign(timestamp_ms, method, path),
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "accept": "application/json",
            "content-type": "application/json",
        }

    # --------------------------------------------------------------- request

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        authenticated: bool = True,
        max_retries: int = 4,
    ) -> dict[str, Any]:
        """Perform an HTTP request with exponential backoff on 429/5xx."""
        # The signature path must include the v2 prefix, exactly as routed.
        path = f"/trade-api/v2{endpoint}"
        url = f"{self.base_url}{endpoint}"
        backoff = 1.0
        last_err: Exception | None = None

        for attempt in range(max_retries):
            try:
                headers = self._auth_headers(method, path) if authenticated else {
                    "accept": "application/json"
                }
                resp = self.session.request(
                    method, url, headers=headers,
                    params=params, json=json_body, timeout=15,
                )
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    log.warning(
                        "Kalshi %s %s returned %s (attempt %d) — backing off %.1fs",
                        method, endpoint, resp.status_code, attempt + 1, backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                if not resp.ok:
                    raise KalshiAPIError(
                        f"{method} {endpoint} failed: {resp.status_code} {resp.text[:300]}"
                    )
                return resp.json() if resp.content else {}
            except (requests.RequestException, KalshiAPIError) as exc:
                last_err = exc
                log.warning("Kalshi request error (attempt %d): %s", attempt + 1, exc)
                time.sleep(backoff)
                backoff *= 2

        raise KalshiAPIError(
            f"Kalshi {method} {endpoint} failed after {max_retries} attempts: {last_err}"
        )

    # ---------------------------------------------------------------- public

    def get_markets(
        self,
        status: str = "open",
        limit: int = 200,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """List markets, optionally filtered by status (``open``/``closed``/``settled``)."""
        params: dict[str, Any] = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._request("GET", "/markets", params=params, authenticated=False)

    def get_all_open_markets(self, max_pages: int = 10) -> list[dict[str, Any]]:
        """Paginate through open markets and return the combined list.

        Kalshi has tens of thousands of open markets, dominated by sports.
        Prefer :meth:`get_markets_by_series` for targeted econ scanning.
        """
        markets: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(max_pages):
            data = self.get_markets(status="open", cursor=cursor)
            markets.extend(data.get("markets", []))
            cursor = data.get("cursor")
            if not cursor:
                break
        return markets

    def get_markets_by_series(
        self,
        series_ticker: str,
        status: str = "open",
        max_pages: int = 8,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Fetch all markets for a single ``series_ticker`` (e.g. ``KXCPI``)."""
        markets: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(max_pages):
            params: dict[str, Any] = {
                "series_ticker": series_ticker,
                "status": status,
                "limit": limit,
            }
            if cursor:
                params["cursor"] = cursor
            data = self._request("GET", "/markets", params=params, authenticated=False)
            markets.extend(data.get("markets", []))
            cursor = data.get("cursor")
            if not cursor:
                break
        return markets

    def get_econ_markets(self, series_tickers: list[str] | None = None) -> list[dict[str, Any]]:
        """Fetch every market across our known econ series in one call."""
        tickers = series_tickers or config.ECON_SERIES_TICKERS
        out: list[dict[str, Any]] = []
        for st in tickers:
            try:
                out.extend(self.get_markets_by_series(st))
            except KalshiAPIError as exc:
                log.warning("Series fetch failed for %s: %s", st, exc)
        return out

    def get_market(self, ticker: str) -> dict[str, Any]:
        """Return a single market by ticker."""
        return self._request("GET", f"/markets/{ticker}", authenticated=False)

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict[str, Any]:
        """Return the current order book for ``ticker``."""
        return self._request(
            "GET",
            f"/markets/{ticker}/orderbook",
            params={"depth": depth},
            authenticated=False,
        )

    # --------------------------------------------------------------- private

    def get_positions(self) -> dict[str, Any]:
        """Return positions held by the authenticated account."""
        return self._request("GET", "/portfolio/positions")

    def place_order(
        self,
        ticker: str,
        side: str,
        action: str,
        count: int,
        price_cents: int,
        client_order_id: str,
        order_type: str = "limit",
    ) -> dict[str, Any]:
        """Submit an order. ``side`` is ``yes``/``no``, ``action`` is ``buy``/``sell``.

        Refuses to send if ``config.SHADOW_MODE`` is on — belt-and-suspenders
        guard so a future code change can't accidentally fire real orders.
        """
        if config.SHADOW_MODE:
            raise KalshiAPIError(
                "SHADOW_MODE is enabled — refusing to place real order. "
                "Set config.SHADOW_MODE = False to go live."
            )
        body = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": order_type,
            "yes_price" if side == "yes" else "no_price": price_cents,
            "client_order_id": client_order_id,
        }
        return self._request("POST", "/portfolio/orders", json_body=body)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an order by its Kalshi order ID."""
        return self._request("DELETE", f"/portfolio/orders/{order_id}")
