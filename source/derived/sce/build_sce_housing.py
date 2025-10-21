import pandas as pd
import janitor
import numpy as np
import json
from pathlib import Path

from source.lib.helpers.process_text import clean_text, clean_date
from source.lib.save_data import save_data

def main():
    INDIR = Path('datastore/raw/sce_housing/data')
    INDIR_CW = Path('datastore/raw/crosswalks/data')
    OUTDIR = Path('output/derived/sce')
    with open('source/lib/config.json', 'r') as f:
        CONFIG = json.load(f)
    YEARS = list(range(CONFIG['SAMPLE_START'], CONFIG['SAMPLE_END'] + 1))

    cw_sce_housing = pd.read_csv(INDIR_CW / "cw_sce_housing.csv")
    sce_housing = clean_sce_housing(cw_sce_housing, indir=INDIR, years=YEARS)
    sce_housing_harmonized = harmonize_date(sce_housing)
    sce_housing_harmonized = harmonize_residence_status_own(sce_housing_harmonized)
    sce_housing_harmonized = harmonize_value_home_at_purchase(sce_housing_harmonized)
    sce_housing_harmonized = harmonize_debt_housing(sce_housing_harmonized)
    sce_housing_harmonized = harmonize_debt_housing_monthly_payment(sce_housing_harmonized)
    sce_housing_harmonized = harmonize_mortgage_rate_on_own_home(sce_housing_harmonized)
    sce_housing_harmonized = harmonize_reason_apply_refi_other(sce_housing_harmonized)
    sce_housing_harmonized = harmonize_likelihood_apply_home_loan_1y_ahead(sce_housing_harmonized)
    sce_housing_harmonized = harmonize_mortgage_rate_self_current(sce_housing_harmonized)
    sce_housing_harmonized = harmonize_mortgage_rate_avg_current(sce_housing_harmonized)
    sce_housing_harmonized = harmonize_credit_score(sce_housing_harmonized)
    sce_housing_harmonized = harmonize_mortgage_rate_type(sce_housing_harmonized)
    sce_housing_harmonized = harmonize_yes_no(sce_housing_harmonized, ['residence_status_own', 'reason_apply_refi_rates', 'reason_apply_refi_other'])
    
    sce_housing_harmonized = (
        sce_housing_harmonized
        .sort_values(by=['date'])
        .select(columns=['sce_id', 'date', 'residence_status_own'] + [col for col in sce_housing_harmonized.columns if col not in ['sce_id', 'date', 'residence_status_own']])
    )
    
    save_data(
        sce_housing_harmonized,
        keys = ['sce_id', 'date'],
        out_file = OUTDIR / 'sce_housing.csv',
        log_file = OUTDIR / 'sce_housing.log',
        sortbykey = True
    )
    
    
def clean_sce_housing(cw_sce_housing, indir=None, years=None):
    dfs = []
    for year in years:
        try:
            rename_map = (
                cw_sce_housing
                .assign(qid = lambda x: clean_text(x['qid']))
                .query('year == @year')
                .query('question_component.notna()')
                .set_index('qid')['question_component']
                .to_dict()
            )
            
            sce_housing_year = (
                pd.read_excel(indir / f"FRBNY-SCE-Housing-Survey-Public-Microdata-Complete.xlsx", sheet_name=f"Data {year}")
                .clean_names()
                .select(columns=list(rename_map.keys()))
                .rename(columns=rename_map)
                .assign(year = year)
            )
            
            dfs.append(sce_housing_year)
            
        except Exception as e:
            print(f"Error processing year {year}:")
            print(e)
            raise
        
    sce_housing = pd.concat(dfs)
    return sce_housing

def harmonize_date(df):
    df['date'] = clean_date(df['date'].astype(str))
    df['date'] = df['date'].fillna(df['year'].astype(str) + '-02-01')
    df['date'] = clean_date(df['date'], aggregation='month')
    df = df.drop(columns=['year'])
    return df

def harmonize_residence_status_own(df):
    df['residence_status_mc'] = df['residence_status_mc'].map({1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2})
    df['residence_status_own'] = df['residence_status_own'].fillna(df['residence_status_mc'])
    df = df.drop(columns=['residence_status_mc'])
    return df

def harmonize_value_home_at_purchase(df):
    df['value_home_at_purchase_cat'] = df['value_home_at_purchase_cat'].map({
        1: 25000, 2: 75000, 3: 125000, 4: 175000, 5: 250000, 6: 400000, 7: 650000, 8: 900000, 9: 1500000
    })
    df['value_home_at_purchase'] = df['value_home_at_purchase_dollars'].fillna(df['value_home_at_purchase_cat'])
    df = df.drop(columns=['value_home_at_purchase_dollars', 'value_home_at_purchase_cat'])
    return df

def harmonize_debt_housing(df):
    df['debt_housing_cat'] = df['debt_housing_cat'].map({
        1: 12500, 2: 37500, 3: 75000, 4: 125000, 5: 175000, 6: 250000, 7: 400000, 8: 650000, 9: 1000000
    })
    df['debt_housing'] = df['debt_housing_dollars'].fillna(df['debt_housing_cat'])
    df = df.drop(columns=['debt_housing_dollars', 'debt_housing_cat'])
    return df

def harmonize_debt_housing_monthly_payment(df):
    df['debt_housing_monthly_payment_cat']=df['debt_housing_monthly_payment_cat'].map({
        1: 250, 2: 750, 3: 1250, 4: 1750, 5: 2250, 6: 2750, 7: 3250, 8: 3750, 9: 4250, 10: 4750, 11: 5250, 12: 5750, 13: 7000
    })
    df['debt_housing_monthly_payment'] = df['debt_housing_monthly_payment_dollars'].fillna(df['debt_housing_monthly_payment_cat'])
    df = df.drop(columns=['debt_housing_monthly_payment_dollars', 'debt_housing_monthly_payment_cat'])
    return df

def harmonize_mortgage_rate_on_own_home(df):
    df['mortgage_rate_on_own_home_cat'] = df['mortgage_rate_on_own_home_cat'].map({
        1: 1, 2: 2.25, 3: 2.75, 4: 3.25, 5: 3.75, 6: 4.25, 7: 4.75, 8: 5.25, 9: 5.75, 10: 6.25, 11: 6.75, 12: 7.25, 13: 7.75, 14: 10
    })
    df['mortgage_rate_on_own_home'] = df['mortgage_rate_on_own_home_dollars'].fillna(df['mortgage_rate_on_own_home_cat'])
    df = df.drop(columns=['mortgage_rate_on_own_home_dollars', 'mortgage_rate_on_own_home_cat'])
    return df

def harmonize_reason_apply_refi_other(df):
    mask = (df['residence_status_own'] == 1) & (df['debt_housing'].notna())
    df.loc[mask, 'reason_apply_refi_other'] = np.where(
        (df.loc[mask, 'reason_apply_refi_increase_balance'] == 1) |
        (df.loc[mask, 'reason_apply_refi_decrease_balance'] == 1) |
        (df.loc[mask, 'reason_apply_refi_increase_term'] == 1) |
        (df.loc[mask, 'reason_apply_refi_reduce_term'] == 1) |
        (df.loc[mask, 'reason_apply_refi_change_to_fixed'] == 1) |
        (df.loc[mask, 'reason_apply_refi_change_to_adjustable'] == 1) |
        (df.loc[mask, 'reason_apply_refi_change_term'] == 1) |
        (df.loc[mask, 'reason_apply_refi_change_servicer'] == 1) |
        (df.loc[mask, 'reason_apply_refi_combine_liens'] == 1) |
        (df.loc[mask, 'reason_apply_refi_consolidate_debt'] == 1) |
        (df.loc[mask, 'reason_apply_refi_other'] == 1),
        1, 2
    )
    df = df.drop(columns=['reason_apply_refi_increase_balance', 'reason_apply_refi_decrease_balance', 'reason_apply_refi_increase_term', 'reason_apply_refi_reduce_term', 'reason_apply_refi_change_to_fixed', 'reason_apply_refi_change_to_adjustable', 'reason_apply_refi_change_term', 'reason_apply_refi_change_servicer', 'reason_apply_refi_combine_liens', 'reason_apply_refi_consolidate_debt'])
    return df

def harmonize_likelihood_apply_home_loan_1y_ahead(df):
    df['likelihood_apply_home_loan_1y_ahead'] = df['likelihood_apply_additional_home_loan_1y_ahead'].fillna(df['likelihood_apply_initial_home_loan_1y_ahead'])
    df = df.drop(columns=['likelihood_apply_additional_home_loan_1y_ahead', 'likelihood_apply_initial_home_loan_1y_ahead'])
    return df

def harmonize_mortgage_rate_self_current(df):
    df['mortgage_rate_self_current_cat'] = df['mortgage_rate_self_current_cat'].map({
        1: 1, 2: 2.25, 3: 2.75, 4: 3.25, 5: 3.75, 6: 4.25, 7: 4.75, 8: 5.25, 9: 5.75, 10: 6.25, 11: 6.75, 12: 7.25, 13: 7.75, 14: 10
    })
    df['mortgage_rate_self_current'] = df['mortgage_rate_self_current_dollars'].fillna(df['mortgage_rate_self_current_cat'])
    df = df.drop(columns=['mortgage_rate_self_current_dollars', 'mortgage_rate_self_current_cat'])
    return df

def harmonize_mortgage_rate_avg_current(df):
    df['mortgage_rate_avg_current_cat'] = df['mortgage_rate_avg_current_cat'].map({
        1: 1, 2: 2.25, 3: 2.75, 4: 3.25, 5: 3.75, 6: 4.25, 7: 4.75, 8: 5.25, 9: 5.75, 10: 6.25, 11: 6.75, 12: 7.25, 13: 7.75, 14: 10
    })
    df['mortgage_rate_avg_current'] = df['mortgage_rate_avg_current_dollars'].fillna(df['mortgage_rate_avg_current_cat'])
    df = df.drop(columns=['mortgage_rate_avg_current_dollars', 'mortgage_rate_avg_current_cat'])
    return df

def harmonize_credit_score(df):
    df['credit_score'] = df['credit_score'].map({
        1: 580,
        2: 625,
        3: 670,
        4: 740,
        5: 800,
        6: pd.NA
    })
    return df

def harmonize_mortgage_rate_type(df):
    df['mortgage_rate_type'] = df['mortgage_rate_type'].map({
        1: 'adjustable',
        2: 'fixed',
        3: pd.NA
    })
    return df

def harmonize_yes_no(df, var_list):
    for var in var_list:
        df[var] = df[var].map({
            1: "yes",
            2: "no"
        })
    return df


if __name__ == "__main__":
    main()


