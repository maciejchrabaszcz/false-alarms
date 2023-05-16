import pandas as pd
from jsonargparse import ArgumentParser


def process_ip(df: pd.DataFrame) -> pd.DataFrame:
    """Splits IP address into 4 columns where each represent different octets of the IP address.

    Args:
        df (pd.DataFrame): dataframe with IP address column.

    Returns:
        pd.DataFrame: Processed dataframe with 4 new columns.
    """
    df["ip_first"] = df["ip"].apply(lambda x: int(x.split(".")[0]))
    df["ip_second"] = df["ip"].apply(lambda x: int(x.split(".")[1]))
    df["ip_third"] = df["ip"].apply(lambda x: int(x.split(".")[2]))
    df["ip_fourth"] = df["ip"].apply(lambda x: int(x.split(".")[3]))
    return df


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--train_file",
        type=str,
        help="Path to train file",
        default="data/cybersecurity_training.csv",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        help="Path to test file",
        default="data/cybersecurity_test.csv",
    )
    parser.add_argument(
        "--output_train_file",
        type=str,
        help="Path to output train file",
        default="data/cybersecurity_training_preprocessed.csv",
    )
    parser.add_argument(
        "--output_test_file",
        type=str,
        help="Path to output test file",
        default="data/cybersecurity_test_preprocessed.csv",
    )
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_file, sep="|")
    test_df = pd.read_csv(args.test_file, sep="|")

    train_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)

    train_df = process_ip(train_df)
    test_df = process_ip(test_df)

    train_df.to_csv(args.output_train_file, sep="|", index=False)
    test_df.to_csv(args.output_test_file, sep="|", index=False)


if __name__ == "__main__":
    main()
