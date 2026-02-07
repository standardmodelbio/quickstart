import random
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from smb_utils import process_ehr_info
except ImportError:
    process_ehr_info = None  # install with: uv pip install "git+https://github.com/standardmodelbio/smb-utils"

# ==========================================
# CHAPTER 1: Setup & Configuration
# ==========================================

MODEL_ID = "standardmodelbio/SMB-v1-1.7B"

# ==========================================
# CHAPTER 2: Create Patient Data
# ==========================================


def create_meds_cohort_with_labels(n_patients=200):
    """
    Generates synthetic MEDS data AND ground truth labels in memory.
    """
    print(f"\n[1/4] Simulating patient data for N={n_patients}...")

    # Define clinical concepts to inject
    CONDITIONS = [
        {
            "code": "ICD10:C34.90",
            "description": "Signal: Lung Cancer",
            "table": "condition",
        },
        {
            "code": "ICD10:J18.9",
            "description": "Noise: Pneumonia",
            "table": "condition",
        },
    ]

    all_events = []
    labels = []

    for i in range(n_patients):
        pid = f"{i:04d}"
        start_date = pd.Timestamp("2023-01-01") + timedelta(days=random.randint(0, 180))
        curr_time = start_date

        # Assign Phenotype (0=Pneumonia, 1-3=Cancer Stages)
        phenotype = np.random.choice([0, 1, 2, 3])
        is_cancer = phenotype > 0

        # --- Generate Labels (Ground Truth) ---
        binary_label = 1 if is_cancer else 0

        if not is_cancer:
            survival_months = np.random.normal(60, 10)  # Healthy-ish
        else:
            # Stage 1 (~24m) -> Stage 3 (~12m)
            survival_months = np.random.normal(30 - (phenotype * 6), 4)

        labels.append(
            {
                "subject_id": pid,
                "readmission_risk": binary_label,
                "phenotype_class": phenotype,
                "overall_survival_months": max(1.0, survival_months),
                "event_observed": 1,
            }
        )

        # --- Generate Events (MEDS Format) ---
        # Everyone gets a standard workup
        all_events.append(
            {
                "subject_id": pid,
                "time": curr_time,
                "code": "CPT:99213",  # Office Visit
                "table": "procedure",
                "value": None,
            }
        )
        curr_time += timedelta(days=2)
        all_events.append(
            {
                "subject_id": pid,
                "time": curr_time,
                "code": "CPT:71260",  # CT Scan
                "table": "procedure",
                "value": None,
            }
        )

        if is_cancer:
            # Inject Signal: Cancer Diagnosis
            curr_time += timedelta(days=5)
            all_events.append(
                {
                    "subject_id": pid,
                    "time": curr_time,
                    "code": CONDITIONS[0]["code"],
                    "table": "condition",
                    "value": None,
                }
            )

            # Inject Signal: Chemo Cycles & Lab Toxicity
            for _ in range(phenotype):
                curr_time += timedelta(days=21)
                all_events.append(
                    {
                        "subject_id": pid,
                        "time": curr_time,
                        "code": "RxNorm:583214",  # Carboplatin
                        "table": "medication",
                        "value": None,
                    }
                )
                # Sicker patients get higher creatinine (kidney stress)
                creat_val = np.random.normal(1.0 + (phenotype * 0.2), 0.1)
                all_events.append(
                    {
                        "subject_id": pid,
                        "time": curr_time,
                        "code": "LOINC:2160-0",
                        "table": "lab",
                        "value": creat_val,
                    }
                )
        else:
            # Inject Noise: Pneumonia Diagnosis
            curr_time += timedelta(days=1)
            all_events.append(
                {
                    "subject_id": pid,
                    "time": curr_time,
                    "code": CONDITIONS[1]["code"],
                    "table": "condition",
                    "value": None,
                }
            )
            curr_time += timedelta(days=14)
            # Normal recovery labs
            all_events.append(
                {
                    "subject_id": pid,
                    "time": curr_time,
                    "code": "LOINC:2160-0",
                    "table": "lab",
                    "value": np.random.normal(0.9, 0.1),
                }
            )

    # Sort to strictly enforce chronological order for Causal LM
    df_meds = (
        pd.DataFrame(all_events)
        .sort_values(["subject_id", "time"])
        .reset_index(drop=True)
    )
    df_labels = pd.DataFrame(labels)

    # Verbose stats
    n_cancer = df_labels["readmission_risk"].sum()
    print(f"   -> Generated {len(df_meds)} total clinical events (MEDS format).")
    print(
        f"   -> Class Balance: {n_cancer} Cancer / {n_patients - n_cancer} Pneumonia."
    )
    return df_meds, df_labels


# ==========================================
# CHAPTER 3: Extract Embeddings
# ==========================================


def extract_embeddings(df, model, tokenizer):
    """
    Passes patient timelines through SMB-v1 to get latent vectors.
    """
    pids = df["subject_id"].unique()
    n_pids = len(pids)
    embeddings = []

    print(f"\n[3/4] Generating embeddings for {n_pids} patients...")
    print("   -> Strategy: Causal Inference (Last Token Pooling)")

    # Define a cutoff to capture the full synthetic history
    end_time = pd.Timestamp("2024-01-01")

    for i, pid in enumerate(pids):
        # Progress indicator every 20 patients
        if (i + 1) % 50 == 0:
            print(f"   -> Processed {i + 1}/{n_pids} patients...")

        # A. Serialize (DataFrame -> String)
        if process_ehr_info is None:
            raise ImportError(
                "smb_utils is required for extract_embeddings. "
                'Install with: uv pip install "git+https://github.com/standardmodelbio/smb-utils"'
            )
        input_text = process_ehr_info(df=df, subject_id=pid, end_time=end_time)

        # B. Tokenize (String -> Tensor)
        # Truncate to 4096 to fit in context window
        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=4096
        ).to(model.device)

        # C. Inference (Tensor -> Hidden States)
        with torch.no_grad():
            outputs = model(inputs.input_ids, output_hidden_states=True)
            # Use the last token to represent the final patient state
            vec = outputs.hidden_states[-1][:, -1, :]
            embeddings.append(vec.cpu())

    print("   -> Inference complete.")
    return torch.cat(embeddings, dim=0)


# ==========================================
# CHAPTER 4: Train Clinical Predictors
# ==========================================


def run_downstream_tasks(X, df_labels):
    """
    Trains standard ML heads on top of the frozen embeddings.
    """
    print("\n[4/4] Training Clinical Task Heads...")

    # 1. Alignment: Ensure row X[i] corresponds to label y[i]
    df_labels = df_labels.sort_values("subject_id").reset_index(drop=True)

    # 2. Splitting: Standard 80/20 train/test split
    X_np = X.numpy()
    train_idx, test_idx = train_test_split(
        df_labels.index, test_size=0.2, random_state=42
    )

    print(f"   -> Split: {len(train_idx)} Train / {len(test_idx)} Test examples.")

    # --- Task 1: Readmission Risk (Binary) ---
    print("\n   --- Task A: Binary Classification (Readmission Risk) ---")
    y_bin = df_labels.loc[train_idx, "readmission_risk"]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_np[train_idx], y_bin)

    y_prob = clf.predict_proba(X_np[test_idx])[:, 1]
    auc = roc_auc_score(df_labels.loc[test_idx, "readmission_risk"], y_prob)
    print(f"   -> ROC-AUC: {auc:.3f}")

    # --- Task 2: Disease Phenotyping (Multiclass) ---
    print("\n   --- Task B: Multiclass Classification (Phenotype Stage) ---")
    y_mc = df_labels.loc[train_idx, "phenotype_class"]
    clf_mc = LogisticRegression(multi_class="multinomial", max_iter=1000)
    clf_mc.fit(X_np[train_idx], y_mc)

    y_pred = clf_mc.predict(X_np[test_idx])
    acc = accuracy_score(df_labels.loc[test_idx, "phenotype_class"], y_pred)
    print(f"   -> Accuracy: {acc:.3f}")

    # --- Task 3: Survival Time (Regression) ---
    print("\n   --- Task C: Regression (Overall Survival Time) ---")
    y_reg = df_labels.loc[train_idx, "overall_survival_months"]
    reg = Ridge(alpha=1.0)
    reg.fit(X_np[train_idx], y_reg)

    y_pred_reg = reg.predict(X_np[test_idx])
    mae = mean_absolute_error(
        df_labels.loc[test_idx, "overall_survival_months"], y_pred_reg
    )
    print(f"   -> MAE: {mae:.2f} months")

    # --- Task 4: Survival Risk (CoxPH) ---
    print("\n   --- Task D: Survival Analysis (Cox Proportional Hazards) ---")
    print("   -> Projecting embeddings to 10D PCA for stability...")
    # Note: CoxPH is unstable on high-dim data with small N.
    # We project embeddings to 10 principal components (PCA) for stability.
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_np)

    cox_df = pd.DataFrame(X_pca, columns=[f"PC{i}" for i in range(10)])
    cox_df["T"] = df_labels["overall_survival_months"]
    cox_df["E"] = df_labels["event_observed"]

    cph = CoxPHFitter()
    cph.fit(cox_df.iloc[train_idx], duration_col="T", event_col="E")

    c_index = cph.score(cox_df.iloc[test_idx], scoring_method="concordance_index")
    print(f"   -> C-Index: {c_index:.3f}")


# ==========================================
# CHAPTER 5: Execution
# ==========================================

if __name__ == "__main__":
    # 1. Generate patient data in MEDS format
    meds_data, labels_data = create_meds_cohort_with_labels(n_patients=200)

    # 2. Load Standard Model
    print("\n[2/4] Loading Standard Model (SMB-v1-1.7B)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, device_map="auto"
    )
    model.eval()

    # 3. Extract patient embeddings
    embeddings = extract_embeddings(meds_data, model, tokenizer)

    # 4. Train clinical task heads on various prediction tasks
    run_downstream_tasks(embeddings, labels_data)
