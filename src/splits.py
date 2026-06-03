import pandas as pd
from sklearn.model_selection import train_test_split


def add_eye_metadata(labels_df: pd.DataFrame) -> pd.DataFrame:
    labels_df = labels_df.copy()

    labels_df['side'] = labels_df['image'].str.extract(r'_(left|right)$')
    labels_df["patient_id"] = labels_df["image"].str.replace(
        r"_(left|right)$",
        "",
        regex=True
    )
    labels_df['binary_label'] = (labels_df['level'] > 0).astype(int)

    return labels_df


def create_patient_level_split(
        labels_df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_seed: int = 42,
) -> pd.DataFrame:
    if round(train_size + val_size + test_size, 6) != 1.0:
        raise ValueError('train_size + val_size + test_size must be equal to 1')
    
    labels_df = add_eye_metadata(labels_df)

    patient_df = (
        labels_df
        .groupby('patient_id')
        .agg(
            max_level=('level', 'max'),
            has_dr=('binary_label', 'max'),
            image_count=('image', 'count')
        )
        .reset_index()
    )

    train_patients, temp_patients = train_test_split(
        patient_df,
        test_size= val_size + test_size,
        random_state=random_seed,
        stratify=patient_df['max_level']
    )

    relative_test_size = test_size / (val_size + test_size)

    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=relative_test_size,
        random_state=random_seed,
        stratify=temp_patients['max_level']
    )

    labels_df['split'] = 'none'

    labels_df.loc[
        labels_df['patient_id'].isin(train_patients['patient_id']),
        'split'
    ] = 'train'

    labels_df.loc[
        labels_df['patient_id'].isin(val_patients['patient_id']),
        'split'
    ] = 'val'


    labels_df.loc[
        labels_df['patient_id'].isin(test_patients['patient_id']),
        'split'
    ] = 'test'

    return labels_df