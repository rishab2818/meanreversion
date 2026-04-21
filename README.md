# Mean Reversion Scanner v5

A local Python stock-scanner app with a built-in browser UI for:

- mean reversion scans
- DCF/value checks
- Monte Carlo risk simulation
- pairs/correlation analysis
- journal tracking
- parameter optimization

The app runs as a small Python web server and serves the UI from `ui.html`.

## Requirements

- Windows, macOS, or Linux
- Python 3.10+ recommended
- Internet connection for live market data

This project currently uses public Yahoo/Stooq data endpoints from Python and does not require an API key.

## Run Locally

1. Open a terminal in the project folder.
2. Start the app:

```bash
python server.py
```

3. Open your browser to:

```text
http://localhost:7432
```

## How It Works

- `server.py` starts the local HTTP server and exposes the API routes.
- `ui.html` is the frontend dashboard.
- `core/` contains the analysis engines, data fetchers, indicators, and storage helpers.

## Project Structure

```text
.
|-- server.py
|-- ui.html
|-- SUGGESTIONS.md
`-- core/
    |-- config.py
    |-- correlation.py
    |-- data.py
    |-- dcf_engine.py
    |-- indicators.py
    |-- ml_optimizer.py
    |-- monte_carlo.py
    |-- mr_engine.py
    `-- storage.py
```

## Notes

- Journal data is stored locally in `journal.json`.
- Saved tuning profiles are stored locally in `stock_profiles.json`.
- The app is currently designed as a local-first tool and not yet hardened for public deployment.

## Troubleshooting

- If the page does not load, make sure `server.py` is still running.
- If live scan results are empty, try again after a few seconds in case the upstream market-data source is slow.
- If port `7432` is already in use, stop the other process or update the port in `core/config.py`.

## Disclaimer

This software is for research and educational use only. It is not financial advice.
