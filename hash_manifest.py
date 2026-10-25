import hashlib
import json
import os


def get_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Читаем файл кусками, чтобы не грузить память
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


manifest = []
filepath = os.path.join("data", "raw", "arrhythmia.data")

if os.path.exists(filepath):
    manifest.append(
        {
            "path": filepath,
            "hash": get_file_hash(filepath),
            "size": os.path.getsize(filepath),
        }
    )

    with open("hash_manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
    print("Манифест создан: hash_manifest.json")
else:
    print(f"ОШИБКА: Файл {filepath} не найден")
