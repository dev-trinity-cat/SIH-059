import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api, predictionToDisplayGrid } from "./api";

const SeaIceContext = createContext(null);

export function SeaIceProvider({ children, fallbackGrid, fallbackSeries }) {
  const [health, setHealth] = useState(null);
  const [status, setStatus] = useState("connecting");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dateRange, setDateRange] = useState({ start: "", end: "" });
  const [targetDate, setTargetDate] = useState("");
  const [forecast, setForecast] = useState(null);
  const [displayGrid, setDisplayGrid] = useState(fallbackGrid);

  const runPredict = useCallback(async (date) => {
    if (!date) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.predict(date);
      setForecast(result);
      const grid = predictionToDisplayGrid(result.prediction);
      if (grid) setDisplayGrid(grid);
      setStatus("live");
    } catch (err) {
      setError(err.message || "Prediction failed");
      setForecast(null);
      setDisplayGrid(fallbackGrid);
      setStatus("mock");
    } finally {
      setLoading(false);
    }
  }, [fallbackGrid]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const h = await api.health();
        if (cancelled) return;
        setHealth(h);
        if (h.ml_service !== "ok") {
          setStatus("mock");
          setError(h.ml_details?.error || "ML service unavailable — showing demo data");
          return;
        }
        const dates = await api.availableDates();
        if (cancelled) return;
        const start = (dates.start || "").slice(0, 10);
        const end = (dates.end || "").slice(0, 10);
        setDateRange({ start, end });
        const initial = end || start;
        setTargetDate(initial);
        setStatus("live");
        if (initial) await runPredict(initial);
      } catch (err) {
        if (cancelled) return;
        setStatus("mock");
        setError(err.message || "API unreachable — showing demo data");
        setDisplayGrid(fallbackGrid);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [fallbackGrid, runPredict]);

  const series = useMemo(() => {
    const meanPct = forecast?.stats?.mean_concentration != null
      ? Math.round(forecast.stats.mean_concentration * 100)
      : null;
    if (meanPct == null) return fallbackSeries;
    return [
      { t: "Forecast", concentration: meanPct },
      { t: "Min", concentration: Math.round((forecast.stats.min_concentration || 0) * 100) },
      { t: "Max", concentration: Math.round((forecast.stats.max_concentration || 0) * 100) },
    ];
  }, [forecast, fallbackSeries]);

  const value = {
    health,
    status,
    error,
    loading,
    dateRange,
    targetDate,
    setTargetDate: (date) => {
      setTargetDate(date);
      runPredict(date);
    },
    forecast,
    displayGrid,
    series,
    refresh: () => runPredict(targetDate),
  };

  return <SeaIceContext.Provider value={value}>{children}</SeaIceContext.Provider>;
}

export function useSeaIce() {
  const ctx = useContext(SeaIceContext);
  if (!ctx) throw new Error("useSeaIce must be used inside SeaIceProvider");
  return ctx;
}
