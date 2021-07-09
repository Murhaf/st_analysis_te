import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.core.series import Series
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

from stopwords import STOP_WORDS_NORWEGIAN


@st.cache(allow_output_mutation=True)
def read_data(file_path):
    data = pd.read_csv(file_path)
    data.drop(['sid'], axis=1, inplace=True)
    data['Categories'] = data['Kategorier'].apply(lambda x: x.split(','))
    return data


@st.cache(allow_output_mutation=True)
def plot_cats(data):
    df = data["Categories"].apply(lambda x: Series(x).value_counts()).sum()
    df = df.items()
    fig = px.pie(df, values=1, names=0)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def plot_kp(data):
    df = data["Kontaktpunkt"].value_counts().items()
    fig = px.pie(df, values=1, names=0)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def get_wordcloud(data, n, word_cloud=False):
    cv = CountVectorizer(max_df=0.7, max_features=n)
    cv.fit_transform(data['sentences'].to_list())

    if word_cloud:
        stopwords = set(STOP_WORDS_NORWEGIAN)
        wc = WordCloud(
            background_color="white",
            stopwords=stopwords,
        ).fit_words(cv.vocabulary_)
        st.image(wc.to_array())
    with st.beta_expander(f'Show top {n} words'):
        sorted_top_words = dict(
            sorted(cv.vocabulary_.items(), key=lambda item: item[1], reverse=True)
        )
        for word, freq in sorted_top_words.items():
            st.markdown(f"- {word} ({int(freq)})")
