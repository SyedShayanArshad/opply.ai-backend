import os
import logging
from twilio.rest import Client

logger = logging.getLogger("app.twilio_service")

def send_whatsapp_alert(to_number: str, message_body: str) -> tuple[bool, str, str | None]:
    """
    Sends a WhatsApp message using the Twilio API.
    Returns (success, sid, error_message)
    """
    try:
        sid = os.getenv("TWILIO_ACCOUNT_SID")
        token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_WHATSAPP_NUMBER")

        if not sid or not token or not from_number:
            logger.error("Twilio credentials not fully configured.")
            return False, "", "Twilio configuration missing"

        client = Client(sid, token)
        
        formatted_to_number = f"whatsapp:{to_number}" if not to_number.startswith("whatsapp:") else to_number
        
        message = client.messages.create(
            from_=from_number,
            body=message_body,
            to=formatted_to_number
        )
        
        logger.info(f"WhatsApp alert sent (SID: {message.sid})")
        return True, message.sid, None
    except Exception as e:
        err_str = str(e)
        if "63038" in err_str or "limit" in err_str.lower():
            logger.warning(f"WhatsApp limit exceeded: {err_str}")
            return False, "", "limit_exceeded"
        else:
            logger.error(f"Failed to send WhatsApp alert: {err_str}")
            return False, "", err_str
