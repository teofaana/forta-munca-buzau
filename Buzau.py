from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df["active_population_persons"] = (
        df["employed_population_persons"] + df["registered_unemployed_persons"]
    )
    df["inactive_population_persons"] = (
        df["working_age_population_15_64_persons"]
        - df["active_population_persons"]
    )

    df["rpo"] = (
        df["employed_population_persons"] / df["resident_population_persons"]
    ) * 100
    df["rae"] = (
        df["active_population_persons"]
        / df["working_age_population_15_64_persons"]
    ) * 100
    df["rorm"] = (
        df["employed_population_persons"] / df["labor_resources_persons"]
    ) * 100
    df["rgo"] = (
        df["employed_population_persons"]
        / df["working_age_population_15_64_persons"]
    ) * 100
    df["rs"] = (
        df["registered_unemployed_persons"] / df["active_population_persons"]
    ) * 100
    df["ri_inact"] = (
        df["inactive_population_persons"] / df["employed_population_persons"]
    ) * 100
    df["ri_pens"] = (
        df["average_pensioners_persons"] / df["employed_population_persons"]
    ) * 100

    return df


def generate_comparative_table_2015_2024(df: pd.DataFrame) -> pd.DataFrame:
    row_2015 = df[df["year"] == 2015].iloc[0]
    row_2024 = df[df["year"] == 2024].iloc[0]

    indicators = [
        {
            "indicator": "populatia rezidenta (pr)",
            "um": "mii persoane",
            "val_2015": row_2015["resident_population_persons"] / 1000,
            "val_2024": row_2024["resident_population_persons"] / 1000,
        },
        {
            "indicator": "populatia ocupata civila (po)",
            "um": "mii persoane",
            "val_2015": row_2015["employed_population_persons"] / 1000,
            "val_2024": row_2024["employed_population_persons"] / 1000,
        },
        {
            "indicator": "someri inregistrati (s)",
            "um": "mii persoane",
            "val_2015": row_2015["registered_unemployed_persons"] / 1000,
            "val_2024": row_2024["registered_unemployed_persons"] / 1000,
        },
        {
            "indicator": "resurse de munca (rm)",
            "um": "mii persoane",
            "val_2015": row_2015["labor_resources_persons"] / 1000,
            "val_2024": row_2024["labor_resources_persons"] / 1000,
        },
        {
            "indicator": "populatia in varsta de munca 15-64 (pvm)",
            "um": "mii persoane",
            "val_2015": (
                row_2015["working_age_population_15_64_persons"] / 1000
            ),
            "val_2024": (
                row_2024["working_age_population_15_64_persons"] / 1000
            ),
        },
        {
            "indicator": "populatia activa (pa = po + s)",
            "um": "mii persoane",
            "val_2015": row_2015["active_population_persons"] / 1000,
            "val_2024": row_2024["active_population_persons"] / 1000,
        },
        {
            "indicator": "rata de ocupare a populatiei rezidente (rpo)",
            "um": "%",
            "val_2015": row_2015["rpo"],
            "val_2024": row_2024["rpo"],
        },
        {
            "indicator": "rata activitatii economice (rae)",
            "um": "%",
            "val_2015": row_2015["rae"],
            "val_2024": row_2024["rae"],
        },
        {
            "indicator": "rata ocuparii resurselor de munca (rorm)",
            "um": "%",
            "val_2015": row_2015["rorm"],
            "val_2024": row_2024["rorm"],
        },
        {
            "indicator": "gradul de ocupare al pvm (rgo)",
            "um": "%",
            "val_2015": row_2015["rgo"],
            "val_2024": row_2024["rgo"],
        },
        {
            "indicator": "rata somajului (rs)",
            "um": "%",
            "val_2015": row_2015["rs"],
            "val_2024": row_2024["rs"],
        },
        {
            "indicator": "rata de intretinere a inactivilor (ri_inact)",
            "um": "la 100 pers.",
            "val_2015": row_2015["ri_inact"],
            "val_2024": row_2024["ri_inact"],
        },
        {
            "indicator": "rata de intretinere a pensionarilor (ri_pens)",
            "um": "la 100 pers.",
            "val_2015": row_2015["ri_pens"],
            "val_2024": row_2024["ri_pens"],
        },
    ]

    records = []
    for item in indicators:
        diff_abs = item["val_2024"] - item["val_2015"]
        diff_pct = (diff_abs / item["val_2015"]) * 100
        records.append(
            {
                "indicator": item["indicator"],
                "u.m.": item["um"],
                "anul 2015": round(item["val_2015"], 2),
                "anul 2024": round(item["val_2024"], 2),
                "modificare absoluta (sau pp)": round(diff_abs, 2),
                "modificare procentuala (%)": round(diff_pct, 2),
            }
        )

    return pd.DataFrame(records)


def plot_comparison_cards(df_comp: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    indicators_to_plot = [
        (
            "rata de ocupare a populatiei rezidente (rpo)",
            "%",
            "rpo_comparatie.png",
        ),
        ("populatia activa (pa = po + s)", "mii pers.", "pa_comparatie.png"),
        ("rata activitatii economice (rae)", "%", "rae_comparatie.png"),
        ("rata ocuparii resurselor de munca (rorm)", "%", "rorm_comparatie.png"),
        ("gradul de ocupare al pvm (rgo)", "%", "rgo_comparatie.png"),
        ("rata somajului (rs)", "%", "rs_comparatie.png"),
        (
            "rata de intretinere a inactivilor (ri_inact)",
            "la 100 pers.",
            "ri_inact_comparatie.png",
        ),
        (
            "rata de intretinere a pensionarilor (ri_pens)",
            "la 100 pers.",
            "ri_pens_comparatie.png",
        ),
    ]

    for ind_name, unit, filename in indicators_to_plot:
        row = df_comp[df_comp["indicator"] == ind_name].iloc[0]
        v15 = row["anul 2015"]
        v24 = row["anul 2024"]
        diff = row["modificare absoluta (sau pp)"]

        diff_label = (
            "pp"
            if "%" in unit
            else ("pers." if "pers." in unit else unit)
        )
        labels = [
            "anul 2015",
            "anul 2024",
            f"modificare 2024 vs 2015\n({diff_label})",
        ]
        values = [v15, v24, diff]
        colors = ["#2b5c8f", "#e68438", "#a61c1c" if diff < 0 else "#2d8a4e"]

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        bars = ax.bar(labels, values, color=colors, width=0.55)

        for bar in bars:
            h = bar.get_height()
            offset = 2.0 if h >= 0 else -12.0
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontweight="bold",
                fontsize=10,
            )

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel(unit, fontsize=11)
        ax.set_title(
            f"{ind_name.upper()}", fontsize=12, pad=12, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()


def plot_multiannual_trends(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150)

    ax = axes[0, 0]
    ax.plot(
        df["year"],
        df["employed_population_persons"] / 1000,
        marker="o",
        color="#1f77b4",
        label="populatie ocupata (mii)",
    )
    ax.set_title(
        "evolutia populatiei ocupate civile (mii persoane)", fontweight="bold"
    )
    ax.set_xticks(df["year"])
    ax.legend()

    ax = axes[0, 1]
    ax.plot(
        df["year"],
        df["rs"],
        marker="s",
        color="#d62728",
        label="rata somajului (%)",
    )
    ax.set_title("evolutia ratei somajului (%)", fontweight="bold")
    ax.set_xticks(df["year"])
    ax.legend()

    ax = axes[1, 0]
    ax.plot(
        df["year"],
        df["rae"],
        marker="^",
        color="#2ca02c",
        label="rata activitatii ec. (rae %)",
    )
    ax.plot(
        df["year"],
        df["rgo"],
        marker="v",
        color="#ff7f0e",
        label="grad de ocupare pvm (rgo %)",
    )
    ax.plot(
        df["year"],
        df["rpo"],
        marker="d",
        color="#9467bd",
        label="rata ocupare pop. rez. (rpo %)",
    )
    ax.set_title("rate de activitate si ocupare (%)", fontweight="bold")
    ax.set_xticks(df["year"])
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.plot(
        df["year"],
        df["ri_inact"],
        marker="o",
        color="#8c564b",
        label="rata dependenta inactivi (la 100 ocupati)",
    )
    ax.plot(
        df["year"],
        df["ri_pens"],
        marker="s",
        color="#e377c2",
        label="rata dependenta pensionari (la 100 ocupati)",
    )
    ax.set_title("rate de intretinere / dependenta economica", fontweight="bold")
    ax.set_xticks(df["year"])
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "multiannual_trends_dashboard.png")
    plt.close()


def find_data_file() -> Path:
    e_path = Path(r"E:\labor_market_buzau_2015_2024.csv")
    if e_path.exists():
        return e_path

    script_dir = Path(__file__).resolve().parent
    local_candidates = [
        script_dir / "labor_market_buzau_2015_2024.csv",
        script_dir / "data" / "labor_market_buzau_2015_2024.csv",
        Path("labor_market_buzau_2015_2024.csv"),
        Path("data/labor_market_buzau_2015_2024.csv"),
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "nu s-a gasit fisierul csv nici la 'E:\\labor_market_buzau_2015_2024.csv' si nici in folderul proiectului."
    )


def main():
    csv_file = find_data_file()
    print(f"folosim datele din: {csv_file.resolve()}\n")

    df = load_and_preprocess_data(str(csv_file))
    df_comp = generate_comparative_table_2015_2024(df)

    print("=== tabel comparativ 2015 vs 2024 ===")
    print(df_comp.to_string(index=False))

    output_dir = Path(__file__).resolve().parent / "figures"
    plot_comparison_cards(df_comp, output_dir)
    plot_multiannual_trends(df, output_dir)

    print(f"\ngraficele sunt gata si s-au salvat in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
