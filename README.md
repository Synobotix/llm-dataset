structure du projet:

llm/
├── pyproject.toml
├── poetry.lock
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   │
│   └── processed/
│       └── .gitkeep
│
├── scripts/
│   ├── test_c4.py
│   └── prepare_c4.py
|   └── clean_c4.py
│
├── src/
│   └── llm/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── cleaning.py
│       │   ├── filtering.py
│       │   └── deduplication.py
│       │
│       ├── tokenizer/
│       │   └── __init__.py
│       │
│       ├── dataset/
│       │   └── __init__.py
│       │
│       └── model/
│           └── __init__.py
│
└── README.md