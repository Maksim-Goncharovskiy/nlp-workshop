# NLP Workshop
## Описание задачи
В ходе воркшопа предстоит решить задачу многоклассовой классификации результатов обратной связи пользователей.

__На входе__ пользовательские ответы на опрос о сервисе Самокат, состоящие из:
- выбор вариантов из списка
- комментарий с произвольным подробным текстом

__Задание:__ предсказать какие тематики из 50-ти возможных затронуты в пользовательских ответах. В одном ответе может затрагиваться сразу несколько тематик.

__Метрика качества__: Точность (доля полностью правильно классифицированных обращений)


## Результаты
### Этапы работы
1. Знакомство с данными, предобработка текстов и тегов, формирование обучающего и валидационного датасетов: см. `./notebooks/Data-Processing/`.
2. Запуск и проверка baseline решения: см. `./notebooks/Research/Baseline.ipynb`.
3. Улучшение baseline решения, перебор различных решений, основанных на классических моделях: см. `./notebooks/Research/Text-Only-Solutions.ipynb`.
4. Использование нейросетевых решений - дообучение bert моделей: см. `./notebooks/Research/Multi-Label-BERT.ipynb`.
5. Получение эмбеддингов текстов при помощи BERT и обучение на них классических моделей: см. `./notebooks/Research/Tiny-Rubert-Embeddings`.
6. Формирование решений на основании лучших найденных решений: см. `.notebooks/Solutions/`.
7. Использование стекинга из простых и показавших хороший результат моделей: см. `.notebooks/Solutions/4-Tfidf-Stacking.ipynb`.
8. Получение решения при помощи `catboost`: см. `.notebooks/Solutions/5-CatBoost.ipynb`.
   
### Лучшее решение

Лучшее решение было получено при помощи стекинга моделей: `.notebooks/Solutions/4-Tfidf-Stacking.ipynb`.

__Результат составил__: 0.3018617021

__Занятое место:__ 65/96 


## Личный прогресс
* Впервые решил задачу __multi-label__ классификации.
* Познакомился с новым инструментом __sklearn__ - `sklearn.multioutput.MultiOutputClassifier`.
* Впервые попробовал стекинг.
* Написал код дообучения BERT для задачи multi-label классификации, оптимизируя функцию `BCEWithLogitsLoss()`.
* Интересно провел время.
