from pathlib import Path
from urllib.request import Request, urlopen
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

URLS = [
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "https://huggingface.co/spaces/20EC0169/customerChurns/raw/main/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]


def download_file(url: str, destination: Path) -> None:
    request = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urlopen(request, timeout=30) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def ensure_data() -> Path:
    """Return the dataset path, downloading it if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 10_000:
        return DATA_PATH

    # If the user already has the CSV in the project root, reuse it.
    root_copy = PROJECT_ROOT / DATA_PATH.name
    if root_copy.exists():
        shutil.copy2(root_copy, DATA_PATH)
        return DATA_PATH

    errors = []
    for url in URLS:
        try:
            download_file(url, DATA_PATH)
            if DATA_PATH.exists() and DATA_PATH.stat().st_size > 10_000:
                print(f"Downloaded dataset to: {DATA_PATH}")
                return DATA_PATH
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            if DATA_PATH.exists():
                DATA_PATH.unlink(missing_ok=True)

    raise RuntimeError(
        "Could not download the dataset automatically.\n"
        "Place WA_Fn-UseC_-Telco-Customer-Churn.csv inside the data/ folder.\n\n"
        + "\n".join(errors)
    )


if __name__ == "__main__":
    path = ensure_data()
    print(path)
