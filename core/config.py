"""Constants, tickers, sector map."""
import os, sys
from datetime import datetime

PORT       = 7432
CACHE_TTL  = 300          # seconds for price data
FUND_TTL   = 60 * 60 * 6  # 6h for fundamentals (slower-moving)
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOURNAL_F  = os.path.join(BASE_DIR, "journal.json")
PROFILES_F = os.path.join(BASE_DIR, "stock_profiles.json")
SCAN_HISTORY_F = os.path.join(BASE_DIR, "scan_history.json")

# CAPM / WACC assumptions (override per-ticker via DCF params)
RISK_FREE         = 0.0425   # 10Y Treasury approx
MKT_PREMIUM       = 0.055    # equity risk premium
CORP_TAX          = 0.21     # US federal
DEFAULT_COST_DEBT = 0.055

def log(m):
    """Unicode-safe logger. Windows consoles default to cp1252 which can't
    encode ✓, ✗, σ, etc. On UnicodeEncodeError we re-encode with replacement."""
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        enc = (getattr(sys.stdout, "encoding", None) or "ascii")
        print(line.encode(enc, errors="replace").decode(enc, errors="replace"), flush=True)

SP500 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","LLY","AVGO",
    "JPM","TSLA","UNH","V","XOM","MA","COST","HD","PG","JNJ",
    "WMT","ABBV","NFLX","BAC","CRM","CVX","MRK","AMD","ORCL","PEP",
    "KO","ADBE","ACN","TMO","LIN","MCD","ABT","CSCO","WFC","IBM",
    "GE","CAT","AXP","BX","ISRG","QCOM","TXN","MS","INTU","SPGI",
    "RTX","GS","DHR","NEE","PFE","AMGN","BKNG","UNP","LOW","BLK",
    "VRTX","SYK","C","SCHW","DE","TJX","CB","GILD","REGN","ADI",
    "MMC","PLD","SBUX","MDT","BMY","LRCX","MO","BSX","SO","ELV",
    "ICE","NOC","SHW","APH","ZTS","CME","CL","WM","AON","MCO",
    "EQIX","ITW","TT","FI","HCA","ETN","GD","PH","EMR","DUK",
    "ADP","NSC","USB","TDG","TGT","NKE","CSX","ECL","COF","FCX",
    "WELL","CTAS","PCAR","MET","MCK","PNC","OKE","AFL","SRE","PSX",
    "AIG","ROP","HLT","MAR","AZO","FDX","APD","EW","ROST","MSCI",
    "DOW","STZ","PAYX","NEM","CARR","ODFL","VLO","F","GM","KMB",
    "MNST","EXC","BDX","PCG","RSG","GWW","CTVA","O","VRSK","HSY",
    "HUM","FAST","KR","CCI","YUM","GEHC","SLB","DLR","WBD","CEG",
    "PEG","ACGL","IR","AME","XEL","WEC","PRU","ALL","OTIS","COR",
    "TROW","IDXX","EA","HIG","PPG","KHC","WY","CLX","KEYS","IQV",
    "MTD","FIS","CDW","VICI","NUE","FANG","BIIB","DVN","HAL","EFX",
    "ROK","AWK","DD","PPL","CBRE","VLTO","CHD","OMC","RMD","ES",
    "STT","RF","CFG","IFF","TSCO","LEN","PHM","APTV","BR","FTV",
    "MPWR","CINF","ETR","AEE","CNP","TRGP","MKC","ZBH","HPQ","TEL",
    "GIS","L","NTRS","DTE","ANSS","SYY","TER","EIX","WAT","STE",
    "CAH","CBOE","K","ULTA","LHX","HBAN","EPAM","AES","IP","BG",
    "SOLV","WST","WRB","WYNN","NDAQ","ROL","LYB","MAS","PKG","TDY",
    "RHI","GL","LKQ","HST","IEX","DRI","NVR","MHK","VFC","FLT",
    "POOL","TRV","HOLX","RJF","HPE","GRMN","JBHT","SWKS","DPZ","ZBRA",
    "EXPD","LNT","CMS","NWSA","NWS","MGM","PFG","WDC","EG","REG",
    "APA","BBWI","MOS","HRL","AIZ","TAP","CF","NRG","JNPR","DVA",
    "IPGP","SEE","ALLE","FMC","CPT","BWA","BEN","IRM","HSIC","HII",
    "WHR","CZR","LNC","MHK","TPR","RL","PVH","VNO","ZION","AAL",
    "DAL","UAL","LUV","CCL","RCL","NCLH","SBAC","AMT","CCI","SPG",
    "PSA","AVB","EQR","MAA","UDR","CPT","ESS","BXP","VTR","PEAK",
    "SMCI","PLTR","MSTR","COIN","RIVN","SOFI","UPST","RBLX","SNAP","UBER",
    "LYFT","ABNB","SHOP","SQ","DKNG","PENN","BYND","LCID","NKLA","GOEV",
]
SP500 = list(dict.fromkeys(SP500))

TOP50_VOL = [
    "NVDA","TSLA","AAPL","AMD","AMZN","MSFT","META","GOOGL","PLTR","SMCI",
    "BAC","F","SOFI","RIVN","MSTR","COIN","SNAP","UBER","LYFT","SQ",
    "ABBV","XOM","JPM","WFC","C","GE","INTC","KO","PFE","T",
    "VZ","CSCO","ORCL","IBM","AAL","DAL","CCL","NCLH","RCL","SHOP",
    "RBLX","DKNG","ABNB","AMC","GME","SIRI","BBY","WBA","CVS","EBAY",
]

YOUR_STOCKS = ["LITE","WOLF","MVST","SMCI","META","GFS","AAOI","NVDA","AMD","TSLA"]

INDIA_LARGECAP = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","BHARTIARTL.NS","INFY.NS",
    "ITC.NS","SBIN.NS","LT.NS","HINDUNILVR.NS","AXISBANK.NS","KOTAKBANK.NS",
    "BAJFINANCE.NS","SUNPHARMA.NS","NTPC.NS","MARUTI.NS","M&M.NS","ULTRACEMCO.NS",
    "TITAN.NS","POWERGRID.NS","ASIANPAINT.NS","HCLTECH.NS","TATAMOTORS.NS","WIPRO.NS",
    "NESTLEIND.NS","ADANIPORTS.NS","TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","COALINDIA.NS",
    "TECHM.NS","CIPLA.NS","BAJAJFINSV.NS","GRASIM.NS","INDUSINDBK.NS","BEL.NS",
    "TRENT.NS","EICHERMOT.NS","HEROMOTOCO.NS","DRREDDY.NS","APOLLOHOSP.NS","DIVISLAB.NS",
    "ADANIENT.NS","ONGC.NS","BPCL.NS","TATACONSUM.NS","BRITANNIA.NS","SHRIRAMFIN.NS",
    "HDFCLIFE.NS","SBILIFE.NS",
]
INDIA_LARGECAP = list(dict.fromkeys(INDIA_LARGECAP))

SECTOR_MAP = {
    "XLK":["AAPL","MSFT","NVDA","AVGO","AMD","ORCL","CSCO","IBM","INTC","QCOM","ADBE","CRM","INTU","TXN","ACN","LRCX","ADI","KEYS","ANSS","CDNS"],
    "XLF":["BRK-B","JPM","V","MA","BAC","WFC","GS","MS","C","AXP","SPGI","MCO","ICE","CME","CB","MET","AFL","PRU","AIG","ALL"],
    "XLV":["UNH","LLY","JNJ","ABBV","MRK","ABT","TMO","DHR","BMY","AMGN","ISRG","GILD","VRTX","REGN","MDT","EW","BSX","SYK","ZTS","IDXX"],
    "XLE":["XOM","CVX","SLB","EOG","PXD","COP","MPC","PSX","VLO","OXY","DVN","HAL","FANG","APA","HES"],
    "XLY":["AMZN","TSLA","HD","MCD","NKE","LOW","SBUX","TJX","BKNG","GM","F","APTV","ROST","TGT","DRI"],
    "XLP":["PG","KO","PEP","WMT","COST","PM","MO","CL","GIS","KMB","HSY","STZ","KHC","CHD","MKC"],
    "XLI":["GE","CAT","RTX","HON","UNP","DE","LMT","NOC","GD","ETN","PH","EMR","ITW","MMM","FDX"],
    "XLC":["META","GOOGL","GOOG","NFLX","DIS","CHTR","T","VZ","TMUS","ATVI","EA","WBD","PARA","NWS"],
    "XLRE":["AMT","PLD","CCI","EQIX","PSA","SPG","O","WY","DLR","VICI","EQR","AVB","BXP","VTR"],
    "XLB":["LIN","APD","SHW","FCX","NEM","DOW","DD","NUE","PPG","ECL","CF","MOS","ALB","FMC"],
    "XLU":["NEE","SO","DUK","EXC","SRE","AEP","D","PCG","PEG","XEL","EIX","AES","WEC","DTE"],
}

def get_sector_etf(ticker):
    for etf, members in SECTOR_MAP.items():
        if ticker in members:
            return etf
    return None
