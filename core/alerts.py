import logging

def send_alert(message, level="INFO"):
    # Console alert (always on)
    print(f"[ALERT-{level}] {message}")

    # Slack / Email hook
    # requests.post(WEBHOOK_URL, json={"text": message})

    logging.log(
        logging.INFO if level == "INFO" else logging.WARNING,
        message
    )
