import requests
def _send_telegram(bot_token: str, chat_id: str, text: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=15)
        return r.ok
    except Exception: return False
def _send_email_sendgrid(api_key: str, from_email: str, to_email: str, subject: str, text: str) -> bool:
    try:
        url = "https://api.sendgrid.com/v3/mail/send"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {"personalizations":[{"to":[{"email":to_email}]}], "from":{"email":from_email}, "subject":subject, "content":[{"type":"text/plain","value":text}]}
        r = requests.post(url, headers=headers, json=data, timeout=15)
        return r.status_code in (200, 202)
    except Exception: return False
def trigger_alerts(df_view, score_min=None, pct_min=None, news_min=None, secrets: dict = None):
    if df_view is None or df_view.empty: return {"count":0,"sent":[]}
    secrets = secrets or {}
    lines = []
    for _, r in df_view.iterrows():
        ok = True
        if score_min is not None and float(r.get("moonshot_score",0)) < float(score_min): ok=False
        if pct_min is not None and float(r.get("pct_change",0)) < float(pct_min): ok=False
        if news_min is not None and int(r.get("news_hits",0)) < int(news_min): ok=False
        if ok:
            sigs = r.get("signals","")
            lines.append(f"{r['symbol']}  price {r.get('price','')}  %chg {round(float(r.get('pct_change',0)),2)}  tags [{sigs}]")
    if not lines: return {"count":0,"sent":[]}
    body = "Moonshot Alerts\n" + "\n".join(lines[:50])
    sent = []
    bot = secrets.get("TELEGRAM_BOT_TOKEN"); chat = secrets.get("TELEGRAM_CHAT_ID")
    if bot and chat:
        if _send_telegram(bot, chat, body): sent.append("telegram")
    sg = secrets.get("SENDGRID_API_KEY"); from_e = secrets.get("EMAIL_FROM"); to_e = secrets.get("EMAIL_TO")
    if sg and from_e and to_e:
        if _send_email_sendgrid(sg, from_e, to_e, "Moonshot Alerts", body): sent.append("sendgrid")
    return {"count":len(lines), "sent":sent, "preview": body[:1000]}
