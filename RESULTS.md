# Example findings

A worked example of what backtesting with this tool looks like — the
strategies tried, what worked, what didn't, and why. All numbers are net
of a 0.3% per-leg commission (0.6% round trip) unless noted, on a
15-ticker MOEX blue-chip universe (SBER, GAZP, LKOH, GMKN, NVTK, ROSN,
TATN, MTSS, MGNT, CHMF, PLZL, ALRS, VTBR, RUAL, AFLT) over a 2-year
window, 10% of equity per position.

## What worked

**Momentum continuation on a large one-day move.** Entry: close up >7%
vs the previous day. Exit: stop-loss 3% / take-profit 8% / max 3 days.

```json
{
  "name": "momentum_7pct_sltp1",
  "universe": ["SBER", "GAZP", "..."],
  "entry": {"type": "pct_change", "field": "close", "lookback_days": 1, "operator": ">", "value": 0.07},
  "exit": {"type": "stop_loss_take_profit", "stop_loss_pct": 0.03, "take_profit_pct": 0.08, "max_days": 3},
  "commission_pct": 0.003
}
```

48 trades / 2 years, **Sharpe 0.80**, +7.48% total return, win rate
60.4%, max drawdown −1.43%. Smaller thresholds (3%, 5%) and plain
fixed-day holds instead of stop/take were all flat-to-negative — the
edge only showed up at a large threshold with active profit-taking, not
as a general "buy any pop" rule.

**Shorting a confirmed trend break.** Entry: 20-day SMA crosses below
50-day SMA. Exit: SMA20 crosses back above SMA50, capped at 60 days.

```json
{
  "name": "trend_death_cross_20_50_short",
  "entry": {"type": "sma_cross", "fast_window": 20, "slow_window": 50, "direction": "below"},
  "exit": {"type": "condition", "condition": {"type": "sma_cross", "fast_window": 20, "slow_window": 50, "direction": "above"}, "max_days": 60},
  "direction": "short",
  "commission_pct": 0.003
}
```

83 trades, **Sharpe 0.31**, +12.64% on blue-chips. Rerun unchanged on a
13-ticker second-tier universe (AFKS, MVID, HYDR, IRAO, MSNG, LSRG, PIKK,
RTKM, RASP, BELU, MTLR, SGZH, OZON): **Sharpe 0.73**, +38.35% — the
*only* strategy of either family that transferred to smaller/less liquid
names, and it transferred better than it started. Its long mirror
(buying a golden cross) did not work on either universe.

## What didn't

- **Mean reversion** — buying dips, fading spikes, RSI oversold/overbought
  (mild or extreme thresholds), in both directions: consistently flat to
  badly negative. Neither "buy the dip" nor "short the rip" earns back
  its commission.
- **Momentum continuation, mirrored to the downside** — shorting a big
  one-day drop expecting continuation: flat to negative across every
  threshold/hold-duration combination tried. Equities' structural upward
  drift and a tendency for sharp drops to see a bounce rather than
  follow-through both cut against it.
- **Trend-following as a general category on a smaller universe** — only
  the one death-cross-short signal transferred; the other four
  SMA-crossover variants tried (golden cross long, 5/20 fast cross,
  50/200 slow cross, an unbounded "let it run" exit) stayed flat-negative
  or got worse. Don't generalize from one strategy's cross-universe
  result to its whole family.
- **Intraday candle-streak scalping** — "N candles of the same color in a
  row → trade in that direction, hold ~10 minutes", swept across 1-min
  and 5-min bars, streak lengths 1–3: all 24 combinations net negative,
  most near −100% (12,000–170,000 trades each). Even at a 0.04%
  commission tariff (Tinkoff Premium, 7.5x cheaper) the best case was
  still −60% — the underlying signal has essentially no edge before
  costs, so a cheaper tariff doesn't rescue it. At this trade frequency,
  gross return (commission-free) is the number that matters first; if
  that's flat, nothing about the fee schedule will fix it.

## Two methodology traps worth knowing before you trust a backtest number

**Position sizing changes the number a lot.** `position_sizing.weight`
defaults to `1.0` — every trade risks 100% of *current* equity,
compounded sequentially. Across hundreds of trades that turns a mild
edge into ±90% swings that have nothing to do with the strategy. Use a
realistic weight (this example uses `0.10`) before trusting a return
number.

**A condition that's a *state*, not an *event*, plus a short
`holding_period`, is a commission-shredding machine.** `price_vs_sma`
and loose thresholds (`rsi < 40`) stay true for many bars in a row; the
engine closes the position after N bars and, if the condition is still
true, reopens immediately. One test case (`price > SMA(10)`, 1-day hold)
generated 2,330 trades and lost 79% purely to round-trip commission,
independent of any real signal. Use an event-like condition (`sma_cross`,
a sharp threshold) or a `holding_period` much longer than the state
typically persists.
