from pathlib import Path

import pandas as pd
from jsonargparse import ArgumentParser
from ydata_profiling import ProfileReport


def main(
    training_data_path: str = "data/cybersecurity_training.csv",
    test_data_path: str = "data/cybersecurity_test.csv",
    localized_alerts_path: str = "data/localized_alerts_data.csv",
    save_folder: str = "reports",
):
    save_folder = Path(save_folder)
    save_folder.mkdir(exist_ok=True, parents=True)
    training_data = pd.read_csv(training_data_path, sep="|")

    profile = ProfileReport(training_data, title="Train Data Report")
    profile.to_file(save_folder / "train_data_report.html")
    del training_data

    test_data = pd.read_csv(test_data_path, sep="|")
    profile = ProfileReport(test_data, title="Test Data Report")
    profile.to_file(save_folder / "test_data_report.html")
    del test_data

    localized_alerts = pd.read_csv(localized_alerts_path, sep="|")
    profile = ProfileReport(
        localized_alerts, title="Localized Alerts Data Report"
    )
    profile.to_file(save_folder / "localized_alerts_data_report.html")


if __name__ == "__main__":
    parser = ArgumentParser("Generate html reports for all data files.")
    parser.add_argument(
        "--training_data_path",
        default="data/cybersecurity_training.csv",
        type=str,
        help="Path to training data.",
    )
    parser.add_argument(
        "--test_data_path",
        default="data/cybersecurity_test.csv",
        type=str,
        help="Path to test data.",
    )
    parser.add_argument(
        "--localized_alerts_path",
        default="data/localized_alerts_data.csv",
        type=str,
        help="Path to localized alerts data.",
    )
    parser.add_argument(
        "--save_folder",
        default="reports",
        type=str,
        help="Folder to which reports will be saved.",
    )
    args = parser.parse_args()
    main(**vars(args))
