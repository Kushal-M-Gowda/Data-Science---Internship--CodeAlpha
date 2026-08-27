import sys
import pandas as pd
import matplotlib.pyplot as plt


def find_column(df, keywords):
    for c in df.columns:
        low = c.lower()
        if any(k in low for k in keywords):
            return c
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python unemployment_analysis.py path/to/unemployment.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    date_col = find_column(df, ["date"])
    rate_col = find_column(df, ["unemployment rate", "unemployed"])
    region_col = find_column(df, ["region", "state", "area"])

    if date_col is None or rate_col is None:
        print("Columns found:", list(df.columns))
        raise ValueError("Could not auto-detect date/rate columns — check column names above.")

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col, rate_col])
    print("Rows after cleaning:", len(df))
    print("Missing values per column:\n", df.isnull().sum())

    # --- Overall trend over time ---
    trend = df.groupby(date_col)[rate_col].mean().sort_index()
    plt.figure(figsize=(10, 5))
    trend.plot()
    plt.title("Average Unemployment Rate Over Time")
    plt.xlabel("Date")
    plt.ylabel(rate_col)
    plt.tight_layout()
    plt.savefig("unemployment_trend.png", dpi=150)
    plt.close()
    print("Saved unemployment_trend.png")

    # --- Region comparison, if a region column exists ---
    if region_col:
        avg_by_region = df.groupby(region_col)[rate_col].mean().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        avg_by_region.plot(kind="bar")
        plt.title(f"Average {rate_col} by {region_col}")
        plt.ylabel(rate_col)
        plt.tight_layout()
        plt.savefig("unemployment_by_region.png", dpi=150)
        plt.close()
        print("Saved unemployment_by_region.png")
        print("\nTop 5 regions by unemployment rate:\n", avg_by_region.head())

    # --- COVID-19 impact: pre vs during lockdown (Mar-Jul 2020) ---
    covid_start = pd.Timestamp("2020-03-01")
    covid_end = pd.Timestamp("2020-07-31")
    pre_covid = df[df[date_col] < covid_start][rate_col].mean()
    during_covid = df[(df[date_col] >= covid_start) & (df[date_col] <= covid_end)][rate_col].mean()
    post_covid = df[df[date_col] > covid_end][rate_col].mean()

    print("\n--- COVID-19 impact ---")
    print(f"Avg rate before Mar 2020: {pre_covid:.2f}")
    print(f"Avg rate during Mar-Jul 2020: {during_covid:.2f}")
    print(f"Avg rate after Jul 2020: {post_covid:.2f}")

    plt.figure(figsize=(6, 5))
    plt.bar(["Pre-COVID", "During COVID (Mar-Jul 2020)", "Post-COVID"],
            [pre_covid, during_covid, post_covid],
            color=["steelblue", "crimson", "seagreen"])
    plt.ylabel(rate_col)
    plt.title("Unemployment Rate: Pre vs During vs Post COVID-19")
    plt.tight_layout()
    plt.savefig("unemployment_covid_impact.png", dpi=150)
    plt.close()
    print("Saved unemployment_covid_impact.png")

    # --- Seasonal pattern: average rate by month ---
    df["month"] = df[date_col].dt.month
    monthly_avg = df.groupby("month")[rate_col].mean()
    plt.figure(figsize=(8, 5))
    monthly_avg.plot(kind="line", marker="o")
    plt.title(f"Average {rate_col} by Month (Seasonality Check)")
    plt.xlabel("Month")
    plt.ylabel(rate_col)
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig("unemployment_seasonality.png", dpi=150)
    plt.close()
    print("Saved unemployment_seasonality.png")

    print("\nKey insights to write up:")
    print("- Compare the pre/during/post COVID averages printed above.")
    print("- Check unemployment_by_region.png for which regions were hit hardest.")
    print("- Check unemployment_seasonality.png for any recurring monthly pattern.")


if __name__ == "__main__":
    main()
