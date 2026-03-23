import csv
import sys
from pathlib import Path


def main() -> int:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("softmax-kernel.csv")

    triton_values = []
    torch_values = []

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError("CSV is empty or missing a header row.")

        required_columns = {"triton", "torch"}
        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"CSV is missing required columns: {', '.join(sorted(missing_columns))}"
            )

        for row in reader:
            triton_values.append(float(row["triton"]))
            torch_values.append(float(row["torch"]))

    if not triton_values or not torch_values:
        raise ValueError("CSV does not contain any data rows.")

    triton_avg = sum(triton_values) / len(triton_values)
    torch_avg = sum(torch_values) / len(torch_values)
    speedup_ratio = (triton_avg - torch_avg) / torch_avg

    print(f"triton average: {triton_avg:.6f}")
    print(f"torch average: {torch_avg:.6f}")
    print(f"triton speedup over torch: {speedup_ratio:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
