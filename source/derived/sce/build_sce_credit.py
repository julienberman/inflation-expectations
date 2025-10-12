import pandas as pd
import janitor
from pathlib import Path

from source.lib.helpers.process_text import clean_date
from source.lib.save_data import save_data

def main():
    INDIR = Path('datastore/raw/sce_credit/data')
    INDIR_CW = Path('datastore/raw/crosswalks/data')
    OUTDIR = Path('datastore/output/derived/sce')
    
    sce_credit = pd.read_excel(
        INDIR / 'FRBNY-SCE-Credit-Access-complete_microdata.xlsx',
        sheet_name='Data',
        skiprows=1
    )
    cw_sce_credit = pd.read_csv(INDIR_CW / 'cw_sce_credit.csv')
    
    sce_credit_clean = clean_sce_credit(sce_credit, cw_sce_credit)
    sce_credit_harmonized = harmonize_likelihood_variables(sce_credit_clean)
    sce_credit_harmonized = harmonize_request_granted_refi(sce_credit_harmonized)
    sce_credit_harmonized = harmonize_reason_no_apply(sce_credit_harmonized)
    sce_credit_harmonized = harmonize_reason_unlikely_apply(sce_credit_harmonized)
    
    save_data(
        sce_credit_harmonized,
        keys = ['sce_id', 'date'],
        out_file = OUTDIR / 'sce_credit.csv',
        log_file = OUTDIR / 'sce_credit.log',
        sortbykey = True
    )

def clean_sce_credit(sce_credit, cw_sce_credit):
    rename_sce_credit = dict(zip(cw_sce_credit['qid'], cw_sce_credit['question']))
    sce_credit_clean = (
        sce_credit
        .rename(columns=rename_sce_credit)
        .clean_names()
        .assign(date = lambda x: clean_date(x['date']))
    )
    return sce_credit_clean

def harmonize_likelihood_variables(df):
    mapping = {1: 10, 2: 30, 3: 50, 4: 70, 5: 90}
    likelihood_vars = [
        "credit_card", "mortgage", "auto_loan",
        "increase_loan_limit", "increase_credit_limit",
        "refi", "student_loan"
    ]

    df_harmonized = (
        df.assign(**{f"likelihood_apply_{v}":df[f"likelihood_apply_{v}_type_a"].map(mapping).fillna(df[f"likelihood_apply_{v}_type_b"])
            for v in likelihood_vars
        })
        .drop(columns=[f"likelihood_apply_{v}_type_a" for v in likelihood_vars] +[f"likelihood_apply_{v}_type_b" for v in likelihood_vars])
    )
    return df_harmonized

def harmonize_request_granted_refi(df):
    mapping = {1: 1, 2: 1, 3: 0}
    df_harmonized = (
        df
        .assign(request_granted_refi = lambda x: x['request_granted_refi_type_a'].map(mapping).fillna(x['request_granted_refi_type_b']))
        .drop(columns=['request_granted_refi_type_a', 'request_granted_refi_type_b'])
    )
    return df_harmonized

def harmonize_reason_no_apply(df):
    reason_no_apply_vars = [
        "credit_card", "mortgage", "auto_loan",
        "increase_loan_limit", "increase_credit_limit",
        "refi", "student_loan"
    ]
    df_harmonized = (
        df
        .assign(**{f"reason_no_apply_{v}_approval": df[f"all_unchecked_reason_no_apply_{v}_approval"].fillna(df[f"some_unchecked_reason_no_apply_{v}_approval"])
            for v in reason_no_apply_vars
        })
        .rename(columns={"all_unchecked_reason_no_apply_other_approval": "reason_no_apply_other_approval"})
        .select(columns=[f"all_unchecked_reason_no_apply_{v}_approval" for v in reason_no_apply_vars] + [f"some_unchecked_reason_no_apply_{v}_approval" for v in reason_no_apply_vars], invert=True)
    )
    return df_harmonized

def harmonize_reason_unlikely_apply(df):
    reason_unlikely_apply_vars = ["satisfied", "time", "knowledge", "rates", "approval"]
    df_harmonized = (
        df.assign(**{
            f"reason_unlikely_apply_{v}":
                (
                    ((df[f"all_unlikely_reason_unlikely_apply_{v}"] == 1) |
                     (df[f"some_unlikely_reason_unlikely_apply_{v}"] == 1))
                    .astype(float)
                )
            for v in reason_unlikely_apply_vars
        })
        .select(columns=[f"all_unlikely_reason_unlikely_apply_{v}" for v in reason_unlikely_apply_vars] + [f"some_unlikely_reason_unlikely_apply_{v}" for v in reason_unlikely_apply_vars], invert=True)
    )

    return df_harmonized


if __name__ == "__main__":
    main()

