import sys
sys.path.append('.')
from pydantic import ValidationError
from configs.lab2_config import Lab2Config

# список плохих конфигураций
cases = [
    {"chunk_size": 0},       # размер чанка не может быть 0
    {"test_size": 1.5},      # тестовая выборка не может быть больше 100%
    {"contamination": -0.1}, # доля аномалий не может быть отрицательной
]

with open("config_negative_test.log", "w", encoding="utf-8") as f:
    for i, case in enumerate(cases, 1):
        try:
            Lab2Config(**case)
            f.write(f"FAIL: {case} был принят\n")
        except ValidationError as e:
            error_msg = e.errors()[0]['msg']
            f.write(f"OK (Тест {i}): {case} отклонен. Ошибка: {error_msg}\n")
            
print("Лог негативных тестов создан: config_negative_test.log")
