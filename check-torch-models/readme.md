# Тестирование дообученных BERT моделей.
## Способы запуска 
### 1) Запуск теста через консоль
`$python main.py` - запустит подсчёт метрики accuracy для модели, описанной в тестовом файле test.json 

### 2) Запуск теста в ноутбуке Test-Book
```python
import main

main.check_torch_model("tiny_rubert", "cointegrated/rubert-tiny")
```
