# IAEML: robotaxi

Курсовой проект IAEML: 2D-среда «robotaxi» на JAX (динамическая bicycle-модель,
пространство наблюдений, функция награды) и обучение агента алгоритмом PPO.

## Отчёты

| Assignment | Тема | Отчёт                                                                          |
|---|---|--------------------------------------------------------------------------------|
| 1. Environment | модель динамики, наблюдения, награда | [`reports/IAEML_Assignment1_report.pdf`](reports/IAEML_Assignment1_report.pdf) |
| 2. ML | PPO, оптимизация вычислений, эксперименты | [`reports/IAEML_Assignment2_report.pdf`](reports/IAEML_Assignment2_report.pdf) |

Рисунки к каждому отчёту лежат в папке этого отчёта. Компилировать два раза
(`pdflatex`), чтобы подставились ссылки.

## Структура

```
.
├── README.md
├── requirements.txt
├── reports/
│   ├── IAEML_Assignment1_report.pdf             отчёт Assignment 1 и его рисунки
│   └── IAEML_Assignment2_report.pdf             отчёт Assignment 2 и его рисунки
├── src/
│   ├── env/
│   │   ├── base.py              базовая среда (из курса)
│   │   ├── tasks.py             карты (из курса)
│   │   └── robotaxi.py          RoboTaxiEnv: динамика, наблюдения, награда
│   ├── utils/
│   │   ├── utils.py, autoreset.py, renderer.py   вспомогательный код (из курса)
│   │   └── scripted_driver.py   скриптовый водитель для демо (Assignment 1)
│   ├── algorithms/
│   │   ├── train_ppo.py         PPO на JAX: обучение (Assignment 2)
│   │   └── eval_ppo.py          оценка политики и запись видео (Assignment 2)
│   ├── demo_robotaxi.py         видео и оценка скриптового водителя (Assignment 1)
│   ├── main.py, main_robotaxi.py   просмотр в pygame
│   └── tests_robotaxi.py        проверки среды (Assignment 1)
├── checkpoints/
│   ├── map0_random_try2/        чекпоинт, конфигурация и лог, карта 0
│   └── map1_random/             то же для карты 1
├── experiments/                 логи неудачных запусков (описаны в отчёте 2)
└── videos/                      видео обоих заданий
```

## Установка

```bash
pip install jax flax optax equinox chex numpy matplotlib
pip install pygame          # только для интерактивного просмотра
pip install imageio-ffmpeg  # для записи видео (или системный ffmpeg)
```

На нативной Windows JAX работает только на CPU.

## Быстрый старт

Assignment 1 (команды из каталога `src/`):

```bash
cd src
python tests_robotaxi.py                                           # проверки среды
python demo_robotaxi.py --map 0 --out ../videos/robotaxi_map0.mp4  # видео скриптового водителя
python main_robotaxi.py                                            # pygame-окно
```

Assignment 2 (команды из корня репозитория):

```bash
# обучение, карта 0 (около 20 минут на CPU)
python src/algorithms/train_ppo.py --map 0 --num-envs 256 --total-timesteps 10000000 --ent-coef 0.003 --out-dir checkpoints/map0_random_try2

# оценка обученной политики и запись видео
python src/algorithms/eval_ppo.py --run-dir checkpoints/map0_random_try2 --map 0 --which best --open
```

Для карты 1: `--map 1` и папка `checkpoints/map1_random`. Остальные запуски
(в том числе неудачные) и их параметры перечислены в разделе «Как воспроизвести
локально» отчёта по Assignment 2.

## Результаты

| Карта | Время до цели (PPO) | Скриптовый водитель |
|---|---|---|
| 0 | 10.8 с | 16.0 с |
| 1 | 17.6 с | 18.3 с |
| 2 | не решена | авария |

Окружение `RoboTaxiEnv` (`src/env/robotaxi.py`) содержит четыре необязательных
параметра, добавленных в Assignment 2 (`replan_period`, `progress_on_path`,
`plan_static_only`, `plug_slits`). По умолчанию они выключены, поэтому поведение
среды из Assignment 1 не меняется.