import csv
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "original"
DOWNLOAD_DIR = PROJECT_ROOT / "dataset" / "kagglehub_download"
DATASET_HANDLE = "dreamer07/eyepacs"


def dataset_is_ready() -> bool:
    labels_path = DATASET_ROOT / "trainLabels.csv"
    images_dir = DATASET_ROOT / "train"
    if not labels_path.is_file() or not images_dir.is_dir():
        return False

    with labels_path.open(encoding="utf-8-sig", newline="") as labels_file:
        reader = csv.DictReader(labels_file)
        if not reader.fieldnames or "image" not in reader.fieldnames:
            return False
        image_names = [row["image"] for row in reader if row.get("image")]

    return bool(image_names) and all(
        (images_dir / f"{image_name}.jpeg").is_file()
        for image_name in image_names
    )


def copy_archive_member(
    archive_path: Path,
    member_name: str,
    destination: Path,
) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member_name) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)


def main() -> None:
    if dataset_is_ready():
        print(f"Dataset already ready: {DATASET_ROOT}")
        return

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is required. Install it with: "
            "python -m pip install kagglehub"
        ) from exc

    download_path = Path(
        kagglehub.dataset_download(
            DATASET_HANDLE,
            output_dir=str(DOWNLOAD_DIR),
        )
    )
    source_files = (
        [download_path]
        if download_path.is_file()
        else [path for path in download_path.rglob("*") if path.is_file()]
    )
    labels_path = next(
        (path for path in source_files if path.name.lower() == "trainlabels.csv"),
        None,
    )
    disk_images = [
        path
        for path in source_files
        if path.suffix.lower() in {".jpg", ".jpeg"}
    ]
    archive_images: list[tuple[Path, str, int]] = []
    archive_labels: tuple[Path, str] | None = None

    for archive_path in source_files:
        if archive_path.suffix.lower() != ".zip":
            continue
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                member_name = PurePosixPath(member.filename).name
                if member_name.lower() == "trainlabels.csv" and archive_labels is None:
                    archive_labels = (archive_path, member.filename)
                elif PurePosixPath(member.filename).suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                }:
                    archive_images.append(
                        (archive_path, member.filename, member.file_size)
                    )

    if labels_path is None and archive_labels is None:
        raise FileNotFoundError(
            f"trainLabels.csv was not found in the downloaded dataset: "
            f"{download_path}"
        )

    if not disk_images and not archive_images:
        raise FileNotFoundError(
            f"No JPEG images were found in the downloaded dataset: "
            f"{download_path}"
        )

    images_dir = DATASET_ROOT / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_destination = DATASET_ROOT / "trainLabels.csv"
    if labels_path is not None:
        if labels_path.resolve() != labels_destination.resolve():
            shutil.move(str(labels_path), str(labels_destination))
    else:
        archive_path, member_name = archive_labels
        copy_archive_member(archive_path, member_name, labels_destination)

    for image_path in disk_images:
        destination = images_dir / image_path.name
        if destination.exists():
            if destination.stat().st_size != image_path.stat().st_size:
                raise FileExistsError(
                    f"Image already exists with different size: {destination}"
                )
            continue
        shutil.move(str(image_path), str(destination))

    for archive_path, member_name, file_size in archive_images:
        destination = images_dir / PurePosixPath(member_name).name
        if destination.exists():
            if destination.stat().st_size != file_size:
                raise FileExistsError(
                    f"Image already exists with different size: {destination}"
                )
            continue
        copy_archive_member(archive_path, member_name, destination)

    print(f"Dataset ready: {DATASET_ROOT}")
    print(f"Labels: {DATASET_ROOT / 'trainLabels.csv'}")
    print(
        f"Images: {images_dir} "
        f"({len(disk_images) + len(archive_images)} found)"
    )


if __name__ == "__main__":
    main()
