import pandas as pd
import matplotlib.pyplot as plt
from project_paths import OUTPUT_ROOT

# Path
RUN_NAME = "base_model"
LOG_FILE = OUTPUT_ROOT / RUN_NAME / "logs" / "train_log.csv"
PLOT_DIR = OUTPUT_ROOT / RUN_NAME / "plots"

if not LOG_FILE.exists():
    raise FileNotFoundError(f"train_log.csv not found: {LOG_FILE}")

PLOT_DIR.mkdir(parents=True, exist_ok=True)

COMBINED_PNG = PLOT_DIR / "base_model_loss.png"
POINTS_CSV = PLOT_DIR / "base_model_loss_points.csv"

# Settings
SMOOTH_WINDOW = 200
SHOW_RAW_TRAIN = False
USE_WIDE_Y_AXIS = True

# Load log
df = pd.read_csv(LOG_FILE)
print("Columns:", df.columns.tolist())


def find_col(candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Column not found. Candidates: {candidates}")


step_col = find_col(["optimizer_step", "global_step", "step"])
train_loss_col = find_col(["train_loss", "loss"])
eval_loss_col = find_col(["eval_loss", "validation_loss", "val_loss"])

# Numeric conversion
df[step_col] = pd.to_numeric(df[step_col], errors="coerce")
df[train_loss_col] = pd.to_numeric(df[train_loss_col], errors="coerce")
df[eval_loss_col] = pd.to_numeric(df[eval_loss_col], errors="coerce")

train_df = df[[step_col, train_loss_col]].dropna().sort_values(step_col).copy()
eval_df = df[[step_col, eval_loss_col]].dropna().sort_values(step_col).copy()

# Smoothing
train_df["smooth_train_loss"] = train_df[train_loss_col].rolling(
    window=SMOOTH_WINDOW,
    min_periods=1,
).mean()

eval_df["smooth_eval_loss"] = eval_df[eval_loss_col].rolling(
    window=3,
    min_periods=1,
).mean()

# Save points
save_df = pd.DataFrame(
    {
        "step": df[step_col],
        "train_loss": df[train_loss_col],
        "eval_loss": df[eval_loss_col],
    }
)
save_df.to_csv(POINTS_CSV, index=False, encoding="utf-8-sig")

# Plot. Base Model Train/Eval Loss
plt.figure(figsize=(10, 5))

if SHOW_RAW_TRAIN:
    plt.plot(
        train_df[step_col],
        train_df[train_loss_col],
        label="Raw Train Loss",
        linewidth=0.4,
        alpha=0.25,
    )

plt.plot(
    train_df[step_col],
    train_df["smooth_train_loss"],
    label="Train Loss",
    linewidth=1.5,
)

plt.plot(
    eval_df[step_col],
    eval_df["smooth_eval_loss"],
    label="Eval Loss",
    marker="o",
    linewidth=1.4,
    markersize=3,
)

plt.xlabel("Optimizer Step")
plt.ylabel("Loss")
plt.title("Base Model Train / Eval Loss")
plt.grid(True, alpha=0.25)
plt.legend()

if USE_WIDE_Y_AXIS:
    plt.ylim(0, 12)

plt.tight_layout()
plt.savefig(COMBINED_PNG, dpi=300)
plt.show()

print(f"Saved combined loss figure: {COMBINED_PNG}")
print(f"Saved loss points csv: {POINTS_CSV}")
