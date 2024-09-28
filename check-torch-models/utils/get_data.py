import pandas as pd


def process_data(text_df, target_df):
    target_cols = [f"trend_id_res{i}" for i in range(50)]

    def make_target_list(row) -> list:
        target_list = []
        for target in target_cols:
            target_list.append(row[target])
        return target_list

    target_df = pd.DataFrame(target_df.apply(make_target_list, axis=1), columns=['target_list'])

    return text_df.fillna(""), target_df


def get_data():
    X_val_path = "../data/processed/val/text_val_df.csv"
    y_val_path = "../data/processed/val/target_val_df.csv"

    X_val = pd.read_csv(X_val_path, index_col=0)
    y_val = pd.read_csv(y_val_path, index_col=0)

    X_val, y_val = process_data(X_val, y_val)

    return X_val, y_val


if __name__ == "__main__":
    X_val, y_val = get_data()
    print(X_val)

