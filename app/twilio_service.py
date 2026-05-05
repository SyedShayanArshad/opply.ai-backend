import os
import logging
from twilio.rest import Client

logger = logging.getLogger("app.twilio_service")

def send_whatsapp_alert(to_number: str, message_body: str) -> str:
    """
    Sends a WhatsApp message using the Twilio API.
    Ensure TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_WHATSAPP_NUMBER are set in .env.
    """
    try:
        sid = os.getenv("TWILIO_ACCOUNT_SID")
        token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_WHATSAPP_NUMBER")

        if not sid or not token or not from_number:
            logger.error("Twilio credentials not fully configured in environment variables.")
            return ""

        client = Client(sid, token)
        
        # WhatsApp numbers must have 'whatsapp:' prefix
        formatted_to_number = f"whatsapp:{to_number}" if not to_number.startswith("whatsapp:") else to_number
        
        message = client.messages.create(
            from_=from_number,
            body=message_body,
            to=formatted_to_number
        )
        
        logger.info(f"WhatsApp alert sent successfully via Twilio (SID: {message.sid})")
        return message.sid
    except Exception as e:
        err_str = str(e)
        if "63038" in err_str or "5 daily messages limit" in err_str:
            logger.warning(f"WhatsApp daily sandbox limit exceeded (5 msgs/day): {err_str}")
        else:
            logger.error(f"Failed to send WhatsApp alert: {err_str}")
        return ""
