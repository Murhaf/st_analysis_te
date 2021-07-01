import itertools

import numpy as np
import streamlit as st

from utils import get_wordcloud, plot_cats, plot_kp, read_data


def main():
    st.title(" 📋 Analysis dashboard")
    uploaded_file = st.file_uploader("File to analyze", type=["csv"])
    if uploaded_file is not None:
        data = read_data(uploaded_file)
    else:
        st.info('Please upload a valid file')
        return

    st.sidebar.header('Filter by')
    kontakt_punkt = st.sidebar.multiselect("Kontaktpunkt", list(data["Kontaktpunkt"].unique()))
    if kontakt_punkt:
        data = data[data["Kontaktpunkt"].isin(kontakt_punkt)]

    categories = list(set([*itertools.chain.from_iterable(data["Categories"])]))
    kategorier = st.sidebar.selectbox("Categories", [""] + categories)
    if kategorier:
        data = data[data["Kategorier"].str.contains(kategorier)]

    years = st.sidebar.multiselect("Year", list(data["Year"].unique()))
    if years:
        data = data[data["Year"].isin(years)]

    input_text = st.sidebar.text_input('Filter using free text', value="")
    if input_text:
        data = data[data['sentences'].str.contains(input_text)]

    if len(data) == 0:
        st.warning('Your fliters returned no results')
        return
    st.write(data)
    st.write(data.describe(exclude=[np.number]))

    if len(kontakt_punkt) != 1:
        st.markdown("**Kontaktpunkt**")
        st.plotly_chart(plot_kp(data))

    st.markdown("**Kategorier**")
    st.plotly_chart(plot_cats(data))

    top_n_words = st.sidebar.slider("Number of most common words to show", 5, 100, 15)
    show_word_cloud = st.checkbox('Show word cloud')
    get_wordcloud(data, top_n_words, show_word_cloud)


if __name__ == "__main__":
    main()
