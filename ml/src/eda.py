"""Exploratory data analysis. Saves plots to ml/reports/eda and a summary JSON."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import NUMERIC, REPORT_DIR, TARGET
from .data_gen import load_or_generate

OUT = REPORT_DIR / "eda"
sns.set_theme(style="whitegrid", palette="viridis")


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=130)
    plt.close()


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    df, source = load_or_generate()
    summary = {"source": source, "rows": len(df), "columns": list(df.columns),
               "missing_pct": (df.isna().mean() * 100).round(2).to_dict(),
               "yield_stats_by_crop": df.groupby("crop")[TARGET].describe().round(3).to_dict("index")}

    # 1. missing values
    miss = df.isna().mean().mul(100).sort_values(ascending=False)
    miss = miss[miss > 0]
    plt.figure(figsize=(7, 3.5)); sns.barplot(x=miss.values, y=miss.index, color="#2f855a")
    plt.xlabel("% missing"); plt.title("Missing values by column"); save("01_missing_values.png")

    # 2. distributions
    cols = NUMERIC[:-1] + [TARGET]
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    for ax, c in zip(axes.ravel(), cols, strict=False):
        sns.histplot(df[c].dropna(), kde=True, ax=ax, color="#2f855a"); ax.set_title(c)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    save("02_distributions.png")

    # 3. correlation heatmap (per-crop standardised target avoids crop-mix confounding in the target column)
    d = df.copy()
    d["yield_rel_to_crop"] = d[TARGET] / d.groupby("crop")[TARGET].transform("median")
    corr = d[NUMERIC + [TARGET, "yield_rel_to_crop"]].corr()
    plt.figure(figsize=(10, 8)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, square=True)
    plt.title("Correlation heatmap (yield_rel_to_crop = yield / crop median)"); save("03_correlation_heatmap.png")
    summary["top_corr_with_relative_yield"] = corr["yield_rel_to_crop"].drop(["yield_rel_to_crop", TARGET]).round(3).sort_values(key=abs, ascending=False).head(6).to_dict()

    # 4. yield vs rainfall / temperature coloured by crop
    for var, fn in [("rainfall_mm", "04_yield_vs_rainfall.png"), ("temperature_c", "05_yield_vs_temperature.png")]:
        plt.figure(figsize=(9, 5.5))
        s = d.sample(3000, random_state=1)
        sns.scatterplot(data=s, x=var, y="yield_rel_to_crop", hue="crop", alpha=.55, s=18, palette="tab10")
        b = d.assign(bin=pd.qcut(d[var], 20, duplicates="drop")).groupby("bin", observed=True).agg(x=(var, "median"), y=("yield_rel_to_crop", "median"))
        plt.plot(b.x, b.y, color="black", lw=2.5, label="binned median")
        plt.ylim(0, 3); plt.title(f"Relative yield vs {var}"); plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        save(fn)

    # 5. yield by crop & season
    plt.figure(figsize=(10, 5)); sns.boxplot(data=df, x="crop", y=TARGET, color="#68d391"); plt.xticks(rotation=35)
    plt.ylim(0, df[TARGET].quantile(.995)); plt.title("Yield distribution by crop"); save("06_yield_by_crop.png")
    plt.figure(figsize=(10, 5)); sns.barplot(data=df, x="crop", y=TARGET, hue="season", errorbar=None); plt.xticks(rotation=35)
    plt.title("Mean yield by crop and season"); save("07_yield_by_crop_season.png")

    # 6. irrigation, soil, trend
    plt.figure(figsize=(8, 4.5)); sns.barplot(data=d, x="irrigation", y="yield_rel_to_crop", hue="season", errorbar=None)
    plt.title("Relative yield by irrigation and season (irrigation matters most in dry seasons)"); save("08_irrigation_effect.png")
    plt.figure(figsize=(8, 4.5)); sns.lineplot(data=d, x="year", y="yield_rel_to_crop", errorbar="ci")
    plt.title("Technology trend: relative yield by year"); save("09_yearly_trend.png")
    plt.figure(figsize=(8, 4.5)); sns.boxplot(data=d, x="soil_type", y="yield_rel_to_crop", color="#9ae6b4"); plt.ylim(0, 3)
    plt.title("Relative yield by soil type"); save("10_soil_effect.png")

    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"EDA complete: {len(list(OUT.glob('*.png')))} plots saved to {OUT}")
    return summary


if __name__ == "__main__":
    run()
