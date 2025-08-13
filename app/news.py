import feedparser
import pandas as pd
BIOTECH_FEEDS = [
    "https://www.biospace.com/rss/",
    "https://www.fiercebiotech.com/rss/xml",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=IBB&region=US&lang=en-US",
    "https://news.google.com/rss/search?q=biotech+when:7d&hl=en-US&gl=US&ceid=US:en"
]
def fetch_biotech_headlines(max_items=80):
    items = []
    for url in BIOTECH_FEEDS:
        try:
            parsed = feedparser.parse(url)
            for e in parsed.entries[:max_items]:
                items.append({"title": getattr(e, "title", ""), "link": getattr(e, "link", ""), "published": getattr(e, "published", ""), "source": parsed.feed.get("title", url)})
        except Exception: pass
    if not items: return pd.DataFrame(columns=["title","link","published","source"])
    df = pd.DataFrame(items).drop_duplicates(subset=["link"]).reset_index(drop=True); return df
def filter_news_for_tickers(df_news: pd.DataFrame, tickers):
    if df_news is None or df_news.empty or not tickers: return df_news
    toks = [f" {t} " for t in tickers]
    mask = df_news["title"].apply(lambda t: any(tok in f" {t} ".upper() for tok in toks))
    return df_news[mask].copy()
