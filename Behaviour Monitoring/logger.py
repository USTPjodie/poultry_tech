"""
logger.py – Behaviour data logging (CSV) and alert dispatching
(email / MQTT / local log file).

Usage:
    from logger import BehaviourLogger, AlertDispatcher
    bl = BehaviourLogger(config.BEHAVIOUR_LOG_CSV)
    bl.write_row(timestamp=..., track_id=..., behaviour=..., ...)

    ad = AlertDispatcher()
    ad.send("Chicken 3 inactive for 25 minutes")
"""

from __future__ import annotations

import csv
import logging
import os
import smtplib
import time
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV Behaviour Logger
# ---------------------------------------------------------------------------

class BehaviourLogger:
    """Append behaviour summary rows to a CSV file.

    The CSV columns are::
        timestamp, track_id, behaviour, confidence,
        cx, cy,        # centroid pixel coordinates
        elapsed_sec    # seconds spent in this behaviour since last row

    Parameters
    ----------
    csv_path : str
        Destination CSV file path. Created with header if it doesn't exist.
    """

    FIELDNAMES = [
        "timestamp", "track_id", "behaviour", "confidence",
        "cx", "cy", "elapsed_sec",
    ]

    def __init__(self, csv_path: str = config.BEHAVIOUR_LOG_CSV):
        self.csv_path = csv_path
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()
            logger.debug("Created behaviour CSV: %s", self.csv_path)

    def write_row(self, **kwargs: Any):
        """Write a single row.  Missing keys default to empty string."""
        row = {k: kwargs.get(k, "") for k in self.FIELDNAMES}
        if not row["timestamp"]:
            row["timestamp"] = datetime.utcnow().isoformat()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(row)

    def write_summary(self, track_summaries: dict[int, dict]):
        """Write one row per track from a summary dictionary.

        Parameters
        ----------
        track_summaries : dict
            Mapping of ``track_id`` → dict with keys matching FIELDNAMES.
        """
        ts = datetime.utcnow().isoformat()
        for tid, info in track_summaries.items():
            self.write_row(
                timestamp=ts,
                track_id=tid,
                behaviour=info.get("behaviour", ""),
                confidence=round(info.get("confidence", 0.0), 3),
                cx=info.get("cx", ""),
                cy=info.get("cy", ""),
                elapsed_sec=round(info.get("elapsed_sec", 0.0), 1),
            )


# ---------------------------------------------------------------------------
# Alert Logger
# ---------------------------------------------------------------------------

class AlertLogger:
    """Append plain-text alerts with timestamps to a log file."""

    def __init__(self, log_path: str = config.ALERT_LOG_FILE):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, message: str):
        ts  = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[{ts}] {message}\n"
        with open(self.log_path, "a") as f:
            f.write(line)
        logger.warning("ALERT: %s", message)


# ---------------------------------------------------------------------------
# Email Alert
# ---------------------------------------------------------------------------

def send_email_alert(subject: str, body: str):
    """Send an alert email via SMTP (configured in config.py).

    Silently logs errors instead of crashing the main loop.
    """
    if not config.EMAIL_ALERTS_ENABLED:
        return
    try:
        msg = MIMEText(body)
        msg["Subject"] = f"[ChickenMonitor] {subject}"
        msg["From"]    = config.EMAIL_SENDER
        msg["To"]      = ", ".join(config.EMAIL_RECIPIENTS)

        with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
            server.sendmail(
                config.EMAIL_SENDER,
                config.EMAIL_RECIPIENTS,
                msg.as_string(),
            )
        logger.info("Email alert sent: %s", subject)
    except Exception as exc:
        logger.error("Failed to send email alert: %s", exc)


# ---------------------------------------------------------------------------
# MQTT Alert
# ---------------------------------------------------------------------------

def send_mqtt_alert(topic: str, payload: str):
    """Publish an alert payload to an MQTT broker.

    Parameters
    ----------
    topic : str
        MQTT topic string.
    payload : str
        Message payload (plain text or JSON string).
    """
    if not config.MQTT_ALERTS_ENABLED:
        return
    try:
        import paho.mqtt.publish as publish  # type: ignore
        publish.single(
            topic,
            payload=payload,
            hostname=config.MQTT_BROKER,
            port=config.MQTT_PORT,
        )
        logger.info("MQTT alert published to %s", topic)
    except Exception as exc:
        logger.error("Failed to publish MQTT alert: %s", exc)


# ---------------------------------------------------------------------------
# Unified Alert Dispatcher
# ---------------------------------------------------------------------------

class AlertDispatcher:
    """Send an alert through all enabled channels (file, email, MQTT).

    Parameters
    ----------
    cooldown_sec : int
        Minimum seconds between identical alerts to prevent spam.
    """

    def __init__(self, cooldown_sec: int = 300):
        self._alert_logger  = AlertLogger()
        self._cooldown      = cooldown_sec
        self._last_sent: dict[str, float] = {}

    def send(self, message: str, subject: str | None = None):
        """Dispatch an alert message.

        Parameters
        ----------
        message : str
            Alert body / payload.
        subject : str | None
            Optional email subject override (defaults to first 80 chars of message).
        """
        now = time.time()
        if now - self._last_sent.get(message, 0) < self._cooldown:
            return  # Duplicate within cooldown window – skip
        self._last_sent[message] = now

        # 1. Write to local file
        self._alert_logger.log(message)

        # 2. Email
        subj = subject or message[:80]
        send_email_alert(subj, message)

        # 3. MQTT
        send_mqtt_alert(config.MQTT_TOPIC_ALERTS, message)


# ---------------------------------------------------------------------------
# Standard Python logging setup helper
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO"):
    """Configure root logger to stdout with timestamp."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
