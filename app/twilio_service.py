import os
import logging
from twilio.rest import Client

logger = logging.getLogger("app.twilio_service")


def send_whatsapp_alert(to_number: str, message_body: str) -> dict:
    """
    Sends a WhatsApp message using the Twilio API.
    Returns a dict: {success: bool, sid: str | None, error_reason: str | None}
    error_reason values: 'daily_limit' | 'not_joined' | 'credentials_missing' | 'unknown' | None
    Ensure TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_WHATSAPP_NUMBER are set in .env.
    """
    try:
        sid = os.getenv("TWILIO_ACCOUNT_SID")
        token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_WHATSAPP_NUMBER")

        if not sid or not token or not from_number:
            logger.error("Twilio credentials not fully configured in environment variables.")
            return {"success": False, "sid": None, "error_reason": "credentials_missing"}

        client = Client(sid, token)

        # WhatsApp numbers must have 'whatsapp:' prefix
        formatted_to_number = f"whatsapp:{to_number}" if not to_number.startswith("whatsapp:") else to_number

        message = client.messages.create(
            from_=from_number,
            body=message_body,
            to=formatted_to_number
        )

        logger.info(f"WhatsApp alert sent successfully via Twilio (SID: {message.sid})")
        return {"success": True, "sid": message.sid, "error_reason": None}

    except Exception as e:
        err_str = str(e)
        if "63038" in err_str or "daily messages limit" in err_str or "5 daily" in err_str:
            logger.warning(f"WhatsApp daily sandbox limit exceeded (5 msgs/day): {err_str}")
            return {"success": False, "sid": None, "error_reason": "daily_limit"}
        elif "21608" in err_str or "not verified" in err_str.lower() or "not joined" in err_str.lower():
            logger.warning(f"WhatsApp number has not joined the sandbox: {err_str}")
            return {"success": False, "sid": None, "error_reason": "not_joined"}
        else:
            logger.error(f"Failed to send WhatsApp alert: {err_str}")
            return {"success": False, "sid": None, "error_reason": "unknown"}
