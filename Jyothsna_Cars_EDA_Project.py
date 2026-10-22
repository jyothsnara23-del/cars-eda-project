"""
Cars Dataset Exploratory Data Analysis (EDA)
Student: Jyothsna R A.
Dataset: UCI Auto MPG
Run: python Jyothsna_Cars_EDA_Project.py
Downloads the dataset, performs data cleaning and EDA, saves six charts,
summary CSVs, and a PDF report.
"""
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eda_outputs"
OUT.mkdir(exist_ok=True)
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
COLS = ["mpg", "cylinders", "displacement", "horsepower", "weight",
        "acceleration", "model_year", "origin", "car_name"]

def load_data():
    with urlopen(DATA_URL, timeout=30) as response:
        lines = response.read().decode("utf-8").splitlines()
    records = []
    for line in lines:
        parts = line.strip().split(maxsplit=8)
        if len(parts) == 9:
            records.append(parts)
    df = pd.DataFrame(records, columns=COLS)
    for col in COLS[:-1]:
        df[col] = pd.to_numeric(df[col].replace("?", np.nan), errors="coerce")
    df["car_name"] = df["car_name"].astype(str).str.strip()
    return df

def save_fig(filename):
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=160, bbox_inches="tight")
    plt.close()

def main():
    print("Downloading and loading UCI Auto MPG dataset...")
    df = load_data()
    rows_before = len(df)
    missing_before = df.isna().sum()
    duplicates_before = int(df.duplicated().sum())
    print("Initial shape:", df.shape)
    print("Data types:\n", df.dtypes)
    print("Missing values:\n", missing_before)
    print("Duplicate rows:", duplicates_before)

    df = df.drop_duplicates().copy()
    # Median imputation is appropriate here for missing horsepower values.
    df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())
    df = df.dropna(subset=["mpg", "weight", "model_year", "origin"]).copy()
    rows_after = len(df)

    print("Rows before cleaning:", rows_before)
    print("Rows after cleaning:", rows_after)
    print("Descriptive statistics:\n", df.describe().round(2))
    df.describe().round(2).to_csv(OUT / "descriptive_statistics.csv")
    missing_before.rename("missing_before_cleaning").to_csv(OUT / "missing_values_before.csv")

    outliers = {}
    for col in ["mpg", "horsepower", "weight", "displacement"]:
        q1, q3 = df[col].quantile([.25, .75])
        iqr = q3 - q1
        outliers[col] = int(((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum())
    pd.Series(outliers, name="potential_outliers").to_csv(OUT / "potential_outliers.csv")

    # 1. MPG histogram
    plt.figure(figsize=(8,5))
    plt.hist(df["mpg"], bins=20, edgecolor="black")
    plt.title("Distribution of Fuel Economy (MPG)")
    plt.xlabel("Miles per gallon"); plt.ylabel("Number of cars")
    save_fig("01_mpg_distribution.png")

    # 2. Cylinder count bar chart
    counts = df["cylinders"].value_counts().sort_index()
    plt.figure(figsize=(8,5))
    plt.bar(counts.index.astype(str), counts.values, edgecolor="black")
    plt.title("Number of Cars by Cylinder Count")
    plt.xlabel("Number of cylinders"); plt.ylabel("Number of cars")
    save_fig("02_cylinder_count.png")

    # 3. MPG by cylinder boxplot
    groups = [df.loc[df["cylinders"] == c, "mpg"].dropna().values
              for c in sorted(df["cylinders"].dropna().unique())]
    labels = [str(c) for c in sorted(df["cylinders"].dropna().unique())]
    plt.figure(figsize=(8,5))
    plt.boxplot(groups, tick_labels=labels)
    plt.title("Fuel Economy by Cylinder Count")
    plt.xlabel("Number of cylinders"); plt.ylabel("Miles per gallon")
    save_fig("03_mpg_by_cylinders.png")

    # 4. Weight vs MPG scatter plot
    plt.figure(figsize=(8,5))
    for c in sorted(df["cylinders"].dropna().unique()):
        part = df[df["cylinders"] == c]
        plt.scatter(part["weight"], part["mpg"], label=f"{int(c)} cylinders", alpha=.7)
    plt.title("Vehicle Weight vs Fuel Economy")
    plt.xlabel("Weight (dataset units)"); plt.ylabel("Miles per gallon")
    plt.legend()
    save_fig("04_weight_vs_mpg.png")

    # 5. Horsepower vs MPG scatter plot
    plt.figure(figsize=(8,5))
    plt.scatter(df["horsepower"], df["mpg"], alpha=.7)
    plt.title("Horsepower vs Fuel Economy")
    plt.xlabel("Horsepower"); plt.ylabel("Miles per gallon")
    save_fig("05_horsepower_vs_mpg.png")

    # 6. Correlation heatmap (matplotlib only)
    numeric = df.select_dtypes(include=np.number)
    corr = numeric.corr()
    plt.figure(figsize=(9,7))
    plt.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation coefficient")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.title("Correlation Between Numerical Features")
    save_fig("06_correlation_heatmap.png")

    correlation = df[["mpg","weight","horsepower","displacement","acceleration"]].corr()["mpg"]
    cyl_means = df.groupby("cylinders")["mpg"].mean().round(2)
    findings = [
        f"The cleaned dataset contains {len(df)} rows and {df.shape[1]} columns.",
        f"Mean fuel economy is {df['mpg'].mean():.2f} MPG; median fuel economy is {df['mpg'].median():.2f} MPG.",
        f"The most common cylinder count is {int(df['cylinders'].mode().iloc[0])}.",
        f"The correlation between vehicle weight and MPG is {correlation['weight']:.2f}.",
        f"The correlation between horsepower and MPG is {correlation['horsepower']:.2f}.",
        "Average MPG by cylinder count: " + "; ".join(f"{int(k)} cylinders: {v:.2f} MPG" for k,v in cyl_means.items()) + ".",
        "Potential outliers flagged by the IQR method (not automatically removed): " + ", ".join(f"{k}: {v}" for k,v in outliers.items()) + "."
    ]
    print("\\nKEY FINDINGS")
    for i, f in enumerate(findings, 1): print(f"{i}. {f}")

    # Create a report from actual dataset results.
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallText", parent=styles["BodyText"], fontSize=8, leading=10))
    story = [
        Spacer(1, 1.5*cm),
        Paragraph("Exploratory Data Analysis on a Cars Dataset", styles["Title"]),
        Spacer(1, .4*cm),
        Paragraph("Student: Jyothsna R A.", styles["Heading2"]),
        Paragraph("Dataset: UCI Auto MPG", styles["BodyText"]),
        Paragraph('Source: <link href="https://archive.ics.uci.edu/dataset/9/auto+mpg">UCI Machine Learning Repository — Auto MPG</link>', styles["SmallText"]),
        Spacer(1, .5*cm),
        Paragraph("1. Objective", styles["Heading1"]),
        Paragraph("This project inspects, cleans, summarizes and visualizes automobile data to identify patterns, relationships and potential outliers. A machine-learning model is not required.", styles["BodyText"]),
        Paragraph("2. Dataset Overview", styles["Heading1"]),
        Paragraph(f"The dataset had {rows_before} rows before cleaning and {len(COLS)} columns. Variables include MPG, cylinders, displacement, horsepower, weight, acceleration, model year, origin and car name.", styles["BodyText"]),
        Paragraph("3. Data Quality and Cleaning", styles["Heading1"]),
        Paragraph(f"Duplicate rows detected before cleaning: {duplicates_before}. Missing values before cleaning: " + ", ".join(f"{k}: {int(v)}" for k,v in missing_before.items() if v) + ".", styles["BodyText"]),
        Paragraph(f"Rows before cleaning: {rows_before}; rows after cleaning: {rows_after}. Missing horsepower values were replaced with the median. Rows missing essential fields (MPG, weight, model year or origin) were removed. Potential outliers were flagged using the IQR method and retained rather than automatically deleted.", styles["BodyText"]),
        Paragraph("4. Descriptive Analysis", styles["Heading1"]),
        Paragraph(f"Mean MPG: {df['mpg'].mean():.2f}; median MPG: {df['mpg'].median():.2f}; standard deviation of MPG: {df['mpg'].std():.2f}. Full descriptive statistics are included in the generated output folder.", styles["BodyText"]),
        PageBreak(),
        Paragraph("5. Visual EDA", styles["Heading1"]),
        Paragraph("Each chart below is generated from the cleaned UCI dataset. Observations should be interpreted from the plotted evidence.", styles["BodyText"]),
    ]
    chart_notes = [
        "Observation: describe the distribution, spread and possible skew of MPG.",
        "Observation: identify the most common cylinder configuration.",
        "Observation: compare median MPG and spread across cylinder groups.",
        "Observation: inspect the relationship between vehicle weight and MPG. Association does not prove causation.",
        "Observation: inspect whether MPG tends to change with horsepower.",
        "Observation: compare the direction and strength of correlations among numerical features."
    ]
    names = ["01_mpg_distribution.png","02_cylinder_count.png","03_mpg_by_cylinders.png",
             "04_weight_vs_mpg.png","05_horsepower_vs_mpg.png","06_correlation_heatmap.png"]
    for i, (name, note) in enumerate(zip(names, chart_notes)):
        story.append(Spacer(1,.2*cm))
        story.append(Image(str(OUT/name), width=15*cm, height=8.2*cm))
        story.append(Paragraph(note, styles["SmallText"]))
        if i in (1,3):
            story.append(PageBreak())
            story.append(Paragraph("5. Visual EDA (continued)", styles["Heading1"]))
    story += [PageBreak(), Paragraph("6. Key Findings", styles["Heading1"])]
    for f in findings:
        story += [Paragraph("• " + f, styles["BodyText"]), Spacer(1,.12*cm)]
    story += [
        Paragraph("7. Limitations", styles["Heading1"]),
        Paragraph("The dataset represents historical vehicle models and does not contain car prices. Results are limited to the available attributes and should not automatically be generalized to all cars. Correlation does not establish causation.", styles["BodyText"]),
        Paragraph("8. Conclusion", styles["Heading1"]),
        Paragraph("The project demonstrates data inspection, cleaning, descriptive statistics, visual analysis and interpretation of automobile attributes. Future analysis could add a separately sourced price dataset or compare fuel economy across model years and origins.", styles["BodyText"]),
        Paragraph("References", styles["Heading1"]),
        Paragraph('UCI Machine Learning Repository, Auto MPG Dataset: <link href="https://archive.ics.uci.edu/dataset/9/auto+mpg">https://archive.ics.uci.edu/dataset/9/auto+mpg</link>', styles["SmallText"])
    ]
    pdf = ROOT / "Jyothsna_Cars_EDA_Report.pdf"
    SimpleDocTemplate(str(pdf), pagesize=A4, rightMargin=1.6*cm, leftMargin=1.6*cm, topMargin=1.4*cm, bottomMargin=1.4*cm).build(story)
    print("\\nPDF report created:", pdf)
    print("Charts and tables saved in:", OUT)

if __name__ == "__main__":
    main()
