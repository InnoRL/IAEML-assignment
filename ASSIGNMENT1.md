# Задание 1: среда автомобиля

Реализация находится в `src/env/robotaxi.py`. `RobotaxiEnv` наследуется от
преподавательского `BaseEnv`. Исходные `src/env/base.py`, карты,
`src/utils/renderer.py`, `src/main.py` и `requirements.txt` сохранены.
Обучения RL в этом этапе нет.

## Запуск

Используйте Python с уже установленными зависимостями из исходного
`requirements.txt`. Команды выполняются из корня `IAEML-assignment`.
Новые обязательные зависимости не добавлены.

```bash
# WASD / стрелки: газ, тормоз, руль влево, руль вправо. Закрытие окна завершает запуск.
JAX_PLATFORMS=cpu python src/assignment1.py --map 1

# Детерминированная последовательность: разгон, свободный ход, торможение, поворот.
JAX_PLATFORMS=cpu python src/assignment1.py --controls-demo --map 1 --steps 420

# Простой контроллер следования пути, без обучения.
JAX_PLATFORMS=cpu python src/assignment1.py --demo --map 1

# Проверки физики и интерфейсов JAX.
PYTHONPATH=src JAX_PLATFORMS=cpu python -m unittest discover -s tests -v
```

Управление: W / вверх задаёт положительное ускорение; S / вниз тормозит;
A / влево поворачивает руль влево; D / вправо — вправо.
Отпущенный газ означает свободный ход с сопротивлением, отпущенный руль
плавно возвращается к нулевому углу. Заднего хода нет.
При столкновении, достижении цели или лимита времени эпизод останавливается.
Для нового эпизода перезапустите программу. В программном интерфейсе используется
`reset(key, params, initial_state)`; автосброс намеренно не скрывает терминальное состояние.

## Видео и воспроизведение

- [Органы управления, 7 секунд](results/controls_demo.mp4).
- [Следование пути до столкновения](results/assignment1_demo.mp4).
- [Технический отчёт](REPORT_ASSIGNMENT1.md).

Для записи нужен **ffmpeg** на `PATH` или уже установленный необязательный
`imageio-ffmpeg`. Обычный запуск и тесты его не требуют.
На этапе выполнения задания пакеты не устанавливались и не обновлялись.

```bash
JAX_PLATFORMS=cpu python src/assignment1.py --controls-demo --headless --map 1 --steps 420 --record results/controls_demo.mp4 --summary results/controls_summary.json
JAX_PLATFORMS=cpu python src/assignment1.py --demo --headless --map 1 --steps 1800 --record results/assignment1_demo.mp4 --summary results/demo_summary.json
```

Физика работает на 60 Гц, видео записывается на 15 кадрах/с. `--headless`
использует SDL dummy display и сохраняет тот же преподавательский рендерер.
Перед сдачей заполните имя/группу, подпись и личную рефлексию в отчёте.
