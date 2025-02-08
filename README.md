# PythonProject

Система за класифициране на ревю на хотел 

- Анализ на статистическите характеристики, описващи данните.
- Проверка за наличие на крайни (екстремални) стойности и тяхна обработка.
- Анализ на зависимостите в набора от данни, илюстриран с подходящи визуализации с
  pandas, matplotlib и seaborn.
- Почистване на текст и извличане на характеристики от него.
- Експериментиране с лематизация и стеминг.
- Експериментиране със Word2Vec и GloVe модели за преобразуване на думи в числови вектори
- Построяване и оценяване на Моделите логистична регресия и наивен Бейсов класификатор. 
  Сравняване на резултатите без и със скалиране и спрямо това дали се използва BoW или TF_IDF
- Сравняване на горните два подхода с невронна мрежа тип Encoder (напр. BERT).
- Потребителски интерфейс със streamlit.

Requeirements to run:

nltk.download('stopwords')

nltk.download('punkt')



To run:
With active venv:
There are 2 ways to run:


1: Models training and evalutains

python main.py

Has options: 
--lemmatization 
--train_w2v 
--build_bert 

train_w2v needs to be called on the first run. 

Runs with options have increased runtime 

Do NOT run with "--build_bert" as it is very memory intensive. This was run in google colab. Results are in png files.



2: User Interface using one of the models in from 1

streamlit run streamlit_app.py
